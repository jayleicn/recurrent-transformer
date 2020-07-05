""" This module will handle the text generation with beam search. """

import torch
import copy
import torch.nn.functional as F

from src.rtransformer.model import RecursiveTransformer, NonRecurTransformer, NonRecurTransformerUntied, TransformerXL
from src.rtransformer.masked_transformer import MTransformer
from src.rtransformer.beam_search import BeamSearch
from src.rtransformer.recursive_caption_dataset import RecursiveCaptionDataset as RCDataset

import logging
logger = logging.getLogger(__name__)


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def mask_tokens_after_eos(input_ids, input_masks,
                          eos_token_id=RCDataset.EOS, pad_token_id=RCDataset.PAD):
    """replace values after `[EOS]` with `[PAD]`,
    used to compute memory for next sentence generation"""
    for row_idx in range(len(input_ids)):
        # possibly more than one `[EOS]`
        cur_eos_idxs = (input_ids[row_idx] == eos_token_id).nonzero()
        if len(cur_eos_idxs) != 0:
            cur_eos_idx = cur_eos_idxs[0, 0].item()
            input_ids[row_idx, cur_eos_idx+1:] = pad_token_id
            input_masks[row_idx, cur_eos_idx+1:] = 0
    return input_ids, input_masks


class Translator(object):
    """Load with trained model and handle the beam search"""
    def __init__(self, opt, checkpoint, model=None):
        self.opt = opt
        self.device = torch.device("cuda" if opt.cuda else "cpu")

        self.model_config = checkpoint["model_cfg"]
        self.max_t_len = self.model_config.max_t_len
        self.max_v_len = self.model_config.max_v_len
        self.num_hidden_layers = self.model_config.num_hidden_layers

        if model is None:
            if opt.recurrent:
                if opt.xl:
                    logger.info("Use recurrent model - TransformerXL")
                    model = TransformerXL(self.model_config).to(self.device)
                else:
                    logger.info("Use recurrent model - Mine")
                    model = RecursiveTransformer(self.model_config).to(self.device)
            else:
                if opt.untied:
                    logger.info("Use untied non-recurrent single sentence model")
                    model = NonRecurTransformerUntied(self.model_config).to(self.device)
                elif opt.mtrans:
                    logger.info("Use masked transformer -- another non-recurrent single sentence model")
                    model = MTransformer(self.model_config).to(self.device)
                else:
                    logger.info("Use non-recurrent single sentence model")
                    model = NonRecurTransformer(self.model_config).to(self.device)
            # model = RecursiveTransformer(self.model_config).to(self.device)
            model.load_state_dict(checkpoint["model"])
        print("[Info] Trained model state loaded.")
        self.model = model
        self.model.eval()

        # self.eval_dataset = eval_dataset

    # @classmethod
    def translate_batch_beam(self, input_ids_list, video_features_list, input_masks_list, token_type_ids_list,
                             rt_model, beam_size, n_best, min_length, max_length, block_ngram_repeat, exclusion_idxs,
                             device, length_penalty_name, length_penalty_alpha):
        # prep the beam object
        base_beam = BeamSearch(
            beam_size,
            n_best=n_best,
            batch_size=len(input_ids_list[0]),
            pad=RCDataset.PAD,
            eos=RCDataset.EOS,
            bos=RCDataset.BOS,
            min_length=min_length,
            max_length=max_length,
            mb_device=device,
            block_ngram_repeat=block_ngram_repeat,
            exclusion_tokens=exclusion_idxs,
            length_penalty_name=length_penalty_name,
            length_penalty_alpha=length_penalty_alpha
        )

        def duplicate_for_beam(prev_ms, input_ids, video_features, input_masks, token_type_ids, beam_size):
            input_ids = tile(input_ids, beam_size, dim=0)  # (N * beam_size, L)
            video_features = tile(video_features, beam_size, dim=0)  # (N * beam_size, L, D_v)
            input_masks = tile(input_masks, beam_size, dim=0)
            token_type_ids = tile(token_type_ids, beam_size, dim=0)
            prev_ms = [tile(e, beam_size, dim=0) for e in prev_ms] \
                if prev_ms[0] is not None else [None] * len(input_ids)
            return prev_ms, input_ids, video_features, input_masks, token_type_ids

        def copy_for_memory(*inputs):
            return [copy.deepcopy(e) for e in inputs]

        def beam_decoding_step(prev_ms, input_ids, video_features, input_masks, token_type_ids, model,
                               max_v_len, max_t_len, beam_size,
                               start_idx=RCDataset.BOS, unk_idx=RCDataset.UNK):
            """
            prev_ms: [(N, M, D), ] * num_hidden_layers or None at first step.
            input_ids: (N, L),
            video_features: (N, L, D_v)
            input_masks: (N, L)
            token_type_ids: (N, L)
            """
            init_ms, init_input_ids, init_video_features, init_input_masks, init_token_type_ids = copy_for_memory(
                prev_ms, input_ids, video_features, input_masks, token_type_ids)

            prev_ms, input_ids, video_features, input_masks, token_type_ids = duplicate_for_beam(
                prev_ms, input_ids, video_features, input_masks, token_type_ids, beam_size=beam_size)

            beam = copy.deepcopy(base_beam)  # copy global variable as local

            # logger.info("batch_size {}, beam_size {}".format(len(input_ids_list[0]), beam_size))
            # logger.info("input_ids {} {}".format(input_ids.shape, input_ids[:6]))
            # logger.info("video_features {}".format(video_features.shape))
            # logger.info("input_masks {} {}".format(input_masks.shape, input_masks[:6]))
            # logger.info("token_type_ids {} {}".format(token_type_ids.shape, token_type_ids[:6]))

            for dec_idx in range(max_v_len, max_v_len + max_t_len):
                # logger.info(" dec_idx {} beam.current_predictions {} {}"
                #             .format(dec_idx, beam.current_predictions.shape, beam.current_predictions))
                input_ids[:, dec_idx] = beam.current_predictions
                input_masks[:, dec_idx] = 1
                copied_prev_ms = copy.deepcopy(prev_ms)  # since the func is changing data inside
                _, _, pred_scores = model.forward_step(
                    copied_prev_ms, input_ids, video_features, input_masks, token_type_ids)
                pred_scores[:, RCDataset.UNK] = -1e10  # remove `[UNK]` token
                logprobs = torch.log(F.softmax(pred_scores[:, dec_idx], dim=1))  # (N * beam_size, vocab_size)
                # next_words = logprobs.max(1)[1]
                # logger.info("next_words {}".format(next_words))
                # import sys
                # sys.exit(1)
                beam.advance(logprobs)
                any_beam_is_finished = beam.is_finished.any()
                if any_beam_is_finished:
                    beam.update_finished()
                    if beam.done:
                        break

                if any_beam_is_finished:
                    # update input args
                    select_indices = beam.current_origin  # N * B, i.e. batch_size * beam_size
                    input_ids = input_ids.index_select(0, select_indices)
                    video_features = video_features.index_select(0, select_indices)
                    input_masks = input_masks.index_select(0, select_indices)
                    token_type_ids = token_type_ids.index_select(0, select_indices)
                    # logger.info("prev_ms {} {}".format(prev_ms[0], type(prev_ms[0])))
                    # logger.info("select_indices {} {}".format(len(select_indices), select_indices))
                    if prev_ms[0] is None:
                        prev_ms = [None] * len(select_indices)
                    else:
                        prev_ms = [e.index_select(0, select_indices) for e in prev_ms]

            # TODO update memory
            # fill in generated words
            for batch_idx in range(len(beam.predictions)):
                cur_sen_ids = beam.predictions[batch_idx][0].cpu().tolist()  # use the top sentences
                cur_sen_ids = [RCDataset.BOS] + cur_sen_ids + [RCDataset.EOS]
                cur_sen_len = len(cur_sen_ids)
                init_input_ids[batch_idx, max_v_len: max_v_len+cur_sen_len] = init_input_ids.new(cur_sen_ids)
                init_input_masks[batch_idx, max_v_len: max_v_len+cur_sen_len] = 1

            # compute memory, mimic the way memory is generated at training time
            init_input_ids, init_input_masks = mask_tokens_after_eos(init_input_ids, init_input_masks)
            cur_ms, _, pred_scores = model.forward_step(
                init_ms, init_input_ids, init_video_features, init_input_masks, init_token_type_ids)

            # logger.info("beam.predictions {}".format(beam.predictions))
            # logger.info("beam.scores {}".format(beam.scores))
            # import sys
            # sys.exit(1)
            # return cur_ms, [e[0][0] for e in beam.predictions]
            return cur_ms, init_input_ids[:, max_v_len:]

        input_ids_list, input_masks_list = self.prepare_video_only_inputs(
            input_ids_list, input_masks_list, token_type_ids_list)
        for cur_input_masks in input_ids_list:
            assert torch.sum(cur_input_masks[:, self.max_v_len + 1:]) == 0, \
                "Initially, all text tokens should be masked."

        config = rt_model.config
        with torch.no_grad():
            prev_ms = [None] * config.num_hidden_layers
            step_size = len(input_ids_list)
            dec_res_list = []
            for idx in range(step_size):
                prev_ms, dec_res = beam_decoding_step(
                    prev_ms, input_ids_list[idx], video_features_list[idx],
                    input_masks_list[idx], token_type_ids_list[idx],
                    rt_model, config.max_v_len, config.max_t_len, beam_size)
                dec_res_list.append(dec_res)
            return dec_res_list

    def translate_batch_greedy(self, input_ids_list, video_features_list, input_masks_list, token_type_ids_list,
                               rt_model):
        def greedy_decoding_step(prev_ms, input_ids, video_features, input_masks, token_type_ids,
                            model, max_v_len, max_t_len, start_idx=RCDataset.BOS, unk_idx=RCDataset.UNK):
            """RTransformer The first few args are the same to the input to the forward_step func
            Note:
                1, Copy the prev_ms each word generation step, as the func will modify this value,
                which will cause discrepancy between training and inference
                2, After finish the current sentence generation step, replace the words generated
                after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
                next memory state tensor.
            """
            bsz = len(input_ids)
            next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
            for dec_idx in range(max_v_len, max_v_len + max_t_len):
                input_ids[:, dec_idx] = next_symbols
                input_masks[:, dec_idx] = 1
                # if dec_idx < max_v_len + 5:
                #     logger.info("prev_ms {} {}".format(type(prev_ms[0]), prev_ms[0]))
                copied_prev_ms = copy.deepcopy(prev_ms)  # since the func is changing data inside
                _, _, pred_scores = model.forward_step(
                    copied_prev_ms, input_ids, video_features, input_masks, token_type_ids)
                # suppress unk token; (N, L, vocab_size)
                pred_scores[:, :, unk_idx] = -1e10
                # next_words = pred_scores.max(2)[1][:, dec_idx]
                next_words = pred_scores[:, dec_idx].max(1)[1]  # TODO / NOTE changed
                next_symbols = next_words

            # compute memory, mimic the way memory is generated at training time
            input_ids, input_masks = mask_tokens_after_eos(input_ids, input_masks)
            cur_ms, _, pred_scores = model.forward_step(
                prev_ms, input_ids, video_features, input_masks, token_type_ids)

            # logger.info("input_ids[:, max_v_len:] {}".format(input_ids[:, max_v_len:]))
            # import sys
            # sys.exit(1)

            return cur_ms, input_ids[:, max_v_len:]  # (N, max_t_len == L-max_v_len)

        input_ids_list, input_masks_list = self.prepare_video_only_inputs(
            input_ids_list, input_masks_list, token_type_ids_list)
        for cur_input_masks in input_ids_list:
            assert torch.sum(cur_input_masks[:, self.max_v_len + 1:]) == 0, \
                "Initially, all text tokens should be masked"

        config = rt_model.config
        with torch.no_grad():
            prev_ms = [None] * config.num_hidden_layers
            step_size = len(input_ids_list)
            dec_seq_list = []
            for idx in range(step_size):
                prev_ms, dec_seq = greedy_decoding_step(
                    prev_ms, input_ids_list[idx], video_features_list[idx],
                    input_masks_list[idx], token_type_ids_list[idx],
                    rt_model, config.max_v_len, config.max_t_len)
                dec_seq_list.append(dec_seq)
            return dec_seq_list

    def translate_batch_greedy_xl(self, input_ids_list, video_features_list, input_masks_list, token_type_ids_list,
                                  rt_model):
        def greedy_decoding_step(prev_ms, input_ids, video_features, token_type_ids, input_masks, prev_masks,
                            model, max_v_len, max_t_len, start_idx=RCDataset.BOS, unk_idx=RCDataset.UNK):
            """TransformerXL: The first few args are the same to the input to the forward_step func
            Note:
                1, Copy the prev_ms each word generation step, as the func will modify this value,
                which will cause discrepancy between training and inference
                2, After finish the current sentence generation step, replace the words generated
                after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
                next memory state tensor.
            """
            bsz = len(input_ids)
            next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
            for dec_idx in range(max_v_len, max_v_len + max_t_len):
                input_ids[:, dec_idx] = next_symbols
                input_masks[:, dec_idx] = 1  # no need to worry about generated <PAD>
                # if dec_idx < max_v_len + 5:
                #     logger.info("prev_ms {} {}".format(type(prev_ms[0]), prev_ms[0]))
                copied_prev_ms = copy.deepcopy(prev_ms)  # since the func is changing data inside
                _, _, pred_scores = model.forward_step(
                    copied_prev_ms, input_ids, video_features, token_type_ids, input_masks, prev_masks)
                # suppress unk token; (N, L, vocab_size)
                pred_scores[:, :, unk_idx] = -1e10
                # next_words = pred_scores.max(2)[1][:, dec_idx]
                next_words = pred_scores[:, dec_idx].max(1)[1]  # TODO / NOTE changed
                next_symbols = next_words

            # compute memory, mimic the way memory is generated at training time
            input_ids, input_masks = mask_tokens_after_eos(input_ids, input_masks)
            cur_ms, _, pred_scores = model.forward_step(
                prev_ms, input_ids, video_features, token_type_ids, input_masks, prev_masks)

            # logger.info("input_ids[:, max_v_len:] {}".format(input_ids[:, max_v_len:]))
            # import sys
            # sys.exit(1)

            return cur_ms, input_ids[:, max_v_len:], input_masks  # (N, max_t_len == L-max_v_len)

        input_ids_list, input_masks_list = self.prepare_video_only_inputs(
            input_ids_list, input_masks_list, token_type_ids_list)
        for cur_input_masks in input_ids_list:
            assert torch.sum(cur_input_masks[:, self.max_v_len + 1:]) == 0, \
                "Initially, all text tokens should be masked"

        config = rt_model.config
        with torch.no_grad():
            prev_ms = rt_model.init_mems()
            step_size = len(input_ids_list)
            dec_seq_list = []
            prev_masks = None
            for idx in range(step_size):
                prev_ms, dec_seq, prev_masks = greedy_decoding_step(
                    prev_ms, input_ids_list[idx], video_features_list[idx],
                    token_type_ids_list[idx], input_masks_list[idx], prev_masks,
                    rt_model, config.max_v_len, config.max_t_len)
                dec_seq_list.append(dec_seq)
            return dec_seq_list

    def translate_batch_single_sentence_greedy(self, input_ids, video_features, input_masks, token_type_ids,
                                               model, start_idx=RCDataset.BOS, unk_idx=RCDataset.UNK):
        """The first few args are the same to the input to the forward_step func
        Note:
            1, Copy the prev_ms each word generation step, as the func will modify this value,
            which will cause discrepancy between training and inference
            2, After finish the current sentence generation step, replace the words generated
            after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
            next memory state tensor.
        """
        input_ids, input_masks = self.prepare_video_only_inputs(input_ids, input_masks, token_type_ids)
        assert torch.sum(input_masks[:, self.max_v_len+1:]) == 0, "Initially, all text tokens should be masked"
        config = model.config
        max_v_len = config.max_v_len
        max_t_len = config.max_t_len
        bsz = len(input_ids)
        next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
        for dec_idx in range(max_v_len, max_v_len + max_t_len):
            input_ids[:, dec_idx] = next_symbols
            input_masks[:, dec_idx] = 1
            # if dec_idx < max_v_len + 5:
            #     logger.info("prev_ms {} {}".format(type(prev_ms[0]), prev_ms[0]))
            _, pred_scores = model.forward(input_ids, video_features, input_masks, token_type_ids, None)
            # suppress unk token; (N, L, vocab_size)
            pred_scores[:, :, unk_idx] = -1e10
            # next_words = pred_scores.max(2)[1][:, dec_idx]
            next_words = pred_scores[:, dec_idx].max(1)[1]  # TODO / NOTE changed
            next_symbols = next_words
        return input_ids[:, max_v_len:]  # (N, max_t_len == L-max_v_len)

    @classmethod
    def translate_batch_single_sentence_untied_greedy(
            cls, video_features, video_masks, text_input_ids, text_masks, text_input_labels,
            model, start_idx=RCDataset.BOS, unk_idx=RCDataset.UNK):
        """The first few args are the same to the input to the forward_step func
        Note:
            1, Copy the prev_ms each word generation step, as the func will modify this value,
            which will cause discrepancy between training and inference
            2, After finish the current sentence generation step, replace the words generated
            after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
            next memory state tensor.
        """
        encoder_outputs = model.encode(video_features, video_masks)  # (N, Lv, D)

        config = model.config
        max_t_len = config.max_t_len
        bsz = len(text_input_ids)
        text_input_ids = text_input_ids.new_zeros(text_input_ids.size())  # all zeros
        text_masks = text_masks.new_zeros(text_masks.size())  # all zeros
        next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
        for dec_idx in range(max_t_len):
            text_input_ids[:, dec_idx] = next_symbols
            text_masks[:, dec_idx] = 1
            _, pred_scores = model.decode(
                text_input_ids, text_masks, text_input_labels, encoder_outputs, video_masks)
            # suppress unk token; (N, L, vocab_size)
            pred_scores[:, :, unk_idx] = -1e10
            # next_words = pred_scores.max(2)[1][:, dec_idx]
            next_words = pred_scores[:, dec_idx].max(1)[1]
            next_symbols = next_words
        return text_input_ids  # (N, Lt)

    def translate_batch(self, model_inputs, use_beam=False, recurrent=True, untied=False, xl=False, mtrans=False):
        """while we used *_list as the input names, they could be non-list for single sentence decoding case"""
        if use_beam:
            if recurrent:
                input_ids_list, video_features_list, input_masks_list, token_type_ids_list = model_inputs
                return self.translate_batch_beam(
                    input_ids_list, video_features_list, input_masks_list, token_type_ids_list,
                    self.model, beam_size=self.opt.beam_size, n_best=self.opt.n_best,
                    min_length=self.opt.min_sen_len, max_length=self.opt.max_sen_len-2,
                    block_ngram_repeat=self.opt.block_ngram_repeat, exclusion_idxs=[], device=self.device,
                    length_penalty_name=self.opt.length_penalty_name,
                    length_penalty_alpha=self.opt.length_penalty_alpha)
            else:
                raise NotImplementedError
        else:
            if recurrent:
                input_ids_list, video_features_list, input_masks_list, token_type_ids_list = model_inputs
                if xl:
                    return self.translate_batch_greedy_xl(
                        input_ids_list, video_features_list, input_masks_list, token_type_ids_list, self.model)
                else:
                    return self.translate_batch_greedy(
                        input_ids_list, video_features_list, input_masks_list, token_type_ids_list, self.model)
            else:  # single sentence
                if untied or mtrans:
                    video_features, video_masks, text_input_ids, text_masks, text_input_labels = model_inputs
                    return self.translate_batch_single_sentence_untied_greedy(
                        video_features, video_masks, text_input_ids, text_masks, text_input_labels, self.model)
                else:
                    input_ids_list, video_features_list, input_masks_list, token_type_ids_list = model_inputs
                    return self.translate_batch_single_sentence_greedy(
                        input_ids_list, video_features_list, input_masks_list, token_type_ids_list,
                        self.model)

    @classmethod
    def prepare_video_only_inputs(cls, input_ids, input_masks, segment_ids):
        """ replace text_ids (except `[BOS]`) in input_ids with `[PAD]` token, for decoding.
        This function is essential!!!
        Args:
            input_ids: (N, L) or [(N, L)] * step_size
            input_masks: (N, L) or [(N, L)] * step_size
            segment_ids: (N, L) or [(N, L)] * step_size
        """
        if isinstance(input_ids, list):
            video_only_input_ids_list = []
            video_only_input_masks_list = []
            for e1, e2, e3 in zip(input_ids, input_masks, segment_ids):
                text_mask = e3 == 1  # text positions (`1`) are replaced
                e1[text_mask] = RCDataset.PAD
                e2[text_mask] = 0  # mark as invalid bits
                video_only_input_ids_list.append(e1)
                video_only_input_masks_list.append(e2)
            return video_only_input_ids_list, video_only_input_masks_list
        else:
            text_mask = segment_ids == 1
            input_ids[text_mask] = RCDataset.PAD
            input_masks[text_mask] = 0
            return input_ids, input_masks
