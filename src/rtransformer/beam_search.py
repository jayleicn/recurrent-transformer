"""
https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam_search.py
"""
import torch

from src.rtransformer.decode_strategy import DecodeStrategy, length_penalty_builder
from src.rtransformer.recursive_caption_dataset import RecursiveCaptionDataset as RCDataset

import logging
logger = logging.getLogger(__name__)


class BeamSearch(DecodeStrategy):
    """Generation beam search.

    Note that the attributes list is not exhaustive. Rather, it highlights
    tensors to document their shape. (Since the state variables' "batch"
    size decreases as beams finish, we denote this axis with a B rather than
    ``batch_size``).

    Args:
        beam_size (int): Number of beams to use (see base ``parallel_paths``).
        batch_size (int): See base.
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        mb_device (torch.device or str): See base ``device``.
        min_length (int): See base.
        max_length (int): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.

    Attributes:
        top_beam_finished (ByteTensor): Shape ``(B,)``.
        _batch_offset (LongTensor): Shape ``(B,)``.
        _beam_offset (LongTensor): Shape ``(batch_size x beam_size,)``.
        alive_seq (LongTensor): See base.
        topk_log_probs (FloatTensor): Shape ``(B x beam_size,)``. These
            are the scores used for the topk operation.
        select_indices (LongTensor or NoneType): Shape
            ``(B x beam_size,)``. This is just a flat view of the
            ``_batch_index``.
        topk_scores (FloatTensor): Shape
            ``(B, beam_size)``. These are the
            scores a sequence will receive if it finishes.
        topk_ids (LongTensor): Shape ``(B, beam_size)``. These are the
            word indices of the topk predictions.
        _batch_index (LongTensor): Shape ``(B, beam_size)``.
        _prev_penalty (FloatTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``.
        _coverage (FloatTensor or NoneType): Shape
            ``(1, B x beam_size, inp_seq_len)``.
        hypotheses (list[list[Tuple[Tensor]]]): Contains a tuple
            of score (float), sequence (long), and attention (float or None).
    """

    def __init__(self, beam_size, batch_size, pad, bos, eos, n_best, mb_device,
                 min_length, max_length, block_ngram_repeat, exclusion_tokens,
                 length_penalty_name=None, length_penalty_alpha=0.):
        super(BeamSearch, self).__init__(
            pad, bos, eos, batch_size, mb_device, beam_size, min_length,
            block_ngram_repeat, exclusion_tokens, max_length)
        # beam parameters
        self.beam_size = beam_size
        self.n_best = n_best
        self.batch_size = batch_size
        self.length_penalty_name = length_penalty_name
        self.length_penalty_func = length_penalty_builder(length_penalty_name)
        self.length_penalty_alpha = length_penalty_alpha

        # result caching
        self.hypotheses = [[] for _ in range(batch_size)]

        # beam state
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        self.best_scores = torch.full([batch_size], -1e10, dtype=torch.float,
                                      device=mb_device)  # (N, )

        self._batch_offset = torch.arange(batch_size, dtype=torch.long)  # (N, )
        self._beam_offset = torch.arange(
            0, batch_size * beam_size, step=beam_size, dtype=torch.long,
            device=mb_device)  # (N, )
        self.topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (beam_size - 1), device=mb_device
        ).repeat(batch_size)  # (B*N), guess: store the current beam probabilities
        self.select_indices = None

        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty((batch_size, beam_size),
                                       dtype=torch.float, device=mb_device)  # (N, B)
        self.topk_ids = torch.empty((batch_size, beam_size), dtype=torch.long,
                                    device=mb_device)  # (N, B)
        self._batch_index = torch.empty([batch_size, beam_size],
                                        dtype=torch.long, device=mb_device)  # (N, B)
        self.done = False
        # "global state" of the old beam
        self._prev_penalty = None
        self._coverage = None

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_origin(self):
        return self.select_indices

    @property
    def current_backptr(self):
        # for testing
        return self.select_indices.view(self.batch_size, self.beam_size)\
            .fmod(self.beam_size)

    def advance(self, log_probs):
        """ current step log_probs (N * B, vocab_size), attn (1, N * B, L)
        Which attention is this??? Guess: the one with the encoder outputs
        """
        vocab_size = log_probs.size(-1)

        # using integer division to get an integer _B without casting
        _B = log_probs.shape[0] // self.beam_size  # batch_size

        # force the output to be longer than self.min_length,
        # by setting prob(EOS) to be a very small number when < min_length
        self.ensure_min_length(log_probs)

        # Multiply probs by the beam probability.
        # logger.info("after log_probs {} {}".format(log_probs.shape, log_probs))
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)
        # logger.info("after log_probs {} {}".format(log_probs.shape, log_probs))

        self.block_ngram_repeats(log_probs)

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token, length_penalty is a float number
        step = len(self)
        length_penalty = self.length_penalty_func(step+1, self.length_penalty_alpha)

        # Flatten probs into a list of possibilities.
        # pick topk in all the paths
        curr_scores = log_probs / length_penalty
        curr_scores = curr_scores.reshape(_B, self.beam_size * vocab_size)
        # self.topk_scores and self.topk_ids => (N, B)
        torch.topk(curr_scores,  self.beam_size, dim=-1,
                   out=(self.topk_scores, self.topk_ids))

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.
        torch.mul(self.topk_scores, length_penalty, out=self.topk_log_probs)

        # Resolve beam origin and map to batch index flat representation.
        torch.div(self.topk_ids, vocab_size, out=self._batch_index)  # _batch_index (N * B)
        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)

        self.topk_ids.fmod_(vocab_size)  # resolve true word ids

        # Append last prediction.
        self.alive_seq = torch.cat(
            [self.alive_seq.index_select(0, self.select_indices),
             self.topk_ids.view(_B * self.beam_size, 1)], -1)  # (N * B, step_size)

        self.is_finished = self.topk_ids.eq(self.eos)  # (N, B)
        self.ensure_max_length()

    def update_finished(self):
        # Penalize beams that finished.
        _B_old = self.topk_log_probs.shape[0]  # batch_size might be changing??? as the beams finished
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance, as we advanced 1 step
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10)
        # on real data (newstest2017) with the pretrained transformer,
        # it's faster to not move this back to the original device
        self.is_finished = self.is_finished.to('cpu')  # (N, B)
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)  # (N, ) initialized as zeros
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)
        non_finished_batch = []
        for i in range(self.is_finished.size(0)):  # (N, )
            b = self._batch_offset[i]  # (0, ..., N-1)
            finished_hyp = self.is_finished[i].nonzero().view(-1)
            # Store finished hypotheses for this batch.
            for j in finished_hyp:
                self.hypotheses[b].append([self.topk_scores[i, j],
                                           predictions[i, j, 1:]])
            # End condition is the top beam finished and we can return
            # n_best hypotheses.
            finish_flag = self.top_beam_finished[i] != 0
            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                best_hyp = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)  # sort by scores
                for n, (score, pred) in enumerate(best_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)
            else:
                non_finished_batch.append(i)
        non_finished = torch.tensor(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]
        # Remove finished batches for the next step. (Not finished beam!!!)
        self.top_beam_finished = self.top_beam_finished.index_select(
            0, non_finished)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        non_finished = non_finished.to(self.topk_ids.device)
        self.topk_log_probs = self.topk_log_probs.index_select(0,
                                                               non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.alive_seq = predictions.index_select(0, non_finished) \
            .view(-1, self.alive_seq.size(-1))
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)
