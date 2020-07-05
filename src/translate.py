""" Translate input text with trained model. """

import os
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import random
import numpy as np
import subprocess
from collections import defaultdict

from src.translator import Translator
from src.rtransformer.recursive_caption_dataset import \
    caption_collate, single_sentence_collate, prepare_batch_inputs
from src.rtransformer.recursive_caption_dataset import RecursiveCaptionDataset as RCDataset
from src.utils import load_json, merge_dicts, save_json


def sort_res(res_dict):
    """res_dict: the submission json entry `results`"""
    final_res_dict = {}
    for k, v in res_dict.items():
        final_res_dict[k] = sorted(v, key=lambda x: float(x["timestamp"][0]))
    return final_res_dict


def run_translate(eval_data_loader, translator, opt):
    # submission template
    batch_res = {"version": "VERSION 1.0",
                 "results": defaultdict(list),
                 "external_data": {"used": "true", "details": "ay"}}
    for raw_batch in tqdm(eval_data_loader, mininterval=2, desc="  - (Translate)"):
        if opt.recurrent:
            # prepare data
            step_sizes = raw_batch[1]  # list(int), len == bsz
            meta = raw_batch[2]  # list(dict), len == bsz
            batch = [prepare_batch_inputs(step_data, device=translator.device)
                     for step_data in raw_batch[0]]
            model_inputs = [
                [e["input_ids"] for e in batch],
                [e["video_feature"] for e in batch],
                [e["input_mask"] for e in batch],
                [e["token_type_ids"] for e in batch]
            ]

            dec_seq_list = translator.translate_batch(
                model_inputs, use_beam=opt.use_beam, recurrent=True, untied=False, xl=opt.xl)

            # example_idx indicates which example is in the batch
            for example_idx, (step_size, cur_meta) in enumerate(zip(step_sizes, meta)):
                # step_idx or we can also call it sen_idx
                for step_idx, step_batch in enumerate(dec_seq_list[:step_size]):
                    batch_res["results"][cur_meta["name"]].append({
                        "sentence": eval_data_loader.dataset.convert_ids_to_sentence(
                            step_batch[example_idx].cpu().tolist()).encode("ascii", "ignore"),
                        "timestamp": cur_meta["timestamp"][step_idx],
                        "gt_sentence": cur_meta["gt_sentence"][step_idx]
                    })
        else:  # single sentence
            meta = raw_batch[2]  # list(dict), len == bsz
            batched_data = prepare_batch_inputs(raw_batch[0], device=translator.device)
            if opt.untied or opt.mtrans:
                model_inputs = [
                    batched_data["video_feature"],
                    batched_data["video_mask"],
                    batched_data["text_ids"],
                    batched_data["text_mask"],
                    batched_data["text_labels"]
                ]
            else:
                model_inputs = [
                    batched_data["input_ids"],
                    batched_data["video_feature"],
                    batched_data["input_mask"],
                    batched_data["token_type_ids"]
                ]

            dec_seq = translator.translate_batch(
                model_inputs, use_beam=opt.use_beam, recurrent=False, untied=opt.untied or opt.mtrans)

            # example_idx indicates which example is in the batch
            for example_idx, (cur_gen_sen, cur_meta) in enumerate(zip(dec_seq, meta)):
                cur_data = {
                    "sentence": eval_data_loader.dataset.convert_ids_to_sentence(
                        cur_gen_sen.cpu().tolist()).encode("ascii", "ignore"),
                    "timestamp": cur_meta["timestamp"],
                    "gt_sentence": cur_meta["gt_sentence"]
                }
                batch_res["results"][cur_meta["name"]].append(cur_data)

        if opt.debug:
            break

    batch_res["results"] = sort_res(batch_res["results"])
    return batch_res


def get_data_loader(opt, eval_mode="val"):
    eval_dataset = RCDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir, video_feature_dir=opt.video_feature_dir,
        duration_file=opt.v_duration_file,
        word2idx_path=opt.word2idx_path, max_t_len=opt.max_t_len,
        max_v_len=opt.max_v_len, max_n_sen=opt.max_n_sen + 10, mode=eval_mode,
        recurrent=opt.recurrent, untied=opt.untied or opt.mtrans)

    if opt.recurrent:  # recurrent model
        collate_fn = caption_collate
    else:  # single sentence
        collate_fn = single_sentence_collate
    eval_data_loader = DataLoader(eval_dataset, collate_fn=collate_fn,
                                  batch_size=opt.batch_size, shuffle=False, num_workers=8)
    return eval_data_loader


def main():
    parser = argparse.ArgumentParser(description="translate.py")

    parser.add_argument("--eval_splits", type=str, nargs="+", default=["val", ],
                        choices=["val", "test"], help="evaluate on val/test set, yc2 only has val")
    parser.add_argument("--res_dir", required=True, help="path to dir containing model .pt file")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")

    # beam search configs
    parser.add_argument("--use_beam", action="store_true", help="use beam search, otherwise greedy search")
    parser.add_argument("--beam_size", type=int, default=2, help="beam size")
    parser.add_argument("--n_best", type=int, default=1, help="stop searching when get n_best from beam search")
    parser.add_argument("--min_sen_len", type=int, default=5, help="minimum length of the decoded sentences")
    parser.add_argument("--max_sen_len", type=int, default=30, help="maximum length of the decoded sentences")
    parser.add_argument("--block_ngram_repeat", type=int, default=0, help="block repetition of ngrams during decoding.")
    parser.add_argument("--length_penalty_name", default="none",
                        choices=["none", "wu", "avg"], help="length penalty to use.")
    parser.add_argument("--length_penalty_alpha", type=float, default=0.,
                        help="Google NMT length penalty parameter (higher = longer generation)")
    parser.add_argument("--eval_tool_dir", type=str, default="./densevid_eval")

    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", default=2019, type=int)
    parser.add_argument("--debug", action="store_true")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    checkpoint = torch.load(os.path.join(opt.res_dir, "model.chkpt"))

    # add some of the train configs
    train_opt = checkpoint["opt"]  # EDict(load_json(os.path.join(opt.res_dir, "model.cfg.json")))
    for k in train_opt.__dict__:
        if k not in opt.__dict__:
            setattr(opt, k, getattr(train_opt, k))
    print("train_opt", train_opt)

    decoding_strategy = "beam{}_lp_{}_la_{}".format(
        opt.beam_size, opt.length_penalty_name, opt.length_penalty_alpha) if opt.use_beam else "greedy"
    save_json(vars(opt),
              os.path.join(opt.res_dir, "{}_eval_cfg.json".format(decoding_strategy)),
              save_pretty=True)

    if opt.dset_name == "anet":
        reference_files_map = {
            "val": [os.path.join(opt.data_dir, e) for e in
                    ["anet_entities_val_1_para.json", "anet_entities_val_2_para.json"]],
            "test": [os.path.join(opt.data_dir, e) for e in
                     ["anet_entities_test_1_para.json", "anet_entities_test_2_para.json"]]}
    else:  # yc2
        reference_files_map = {"val": [os.path.join(opt.data_dir, "yc2_val_anet_format_para.json")]}
    for eval_mode in opt.eval_splits:
        print("Start evaluating {}".format(eval_mode))
        # add 10 at max_n_sen to make the inference stage use all the segments
        eval_data_loader = get_data_loader(opt, eval_mode=eval_mode)
        eval_references = reference_files_map[eval_mode]

        # setup model
        translator = Translator(opt, checkpoint)

        pred_file = os.path.join(opt.res_dir, "{}_pred_{}.json".format(decoding_strategy, eval_mode))
        pred_file = os.path.abspath(pred_file)
        if not os.path.exists(pred_file):
            json_res = run_translate(eval_data_loader, translator, opt=opt)
            save_json(json_res, pred_file, save_pretty=True)
        else:
            print("Using existing prediction file at {}".format(pred_file))

        # COCO language evaluation
        lang_file = pred_file.replace(".json", "_lang.json")
        eval_command = ["python", "para-evaluate.py", "-s", pred_file, "-o", lang_file,
                        "-v", "-r"] + eval_references
        subprocess.call(eval_command, cwd=opt.eval_tool_dir)

        # basic stats
        stat_filepath = pred_file.replace(".json", "_stat.json")
        eval_stat_cmd = ["python", "get_caption_stat.py", "-s", pred_file, "-r", eval_references[0],
                         "-o", stat_filepath, "-v"]
        subprocess.call(eval_stat_cmd, cwd=opt.eval_tool_dir)

        # repetition evaluation
        rep_filepath = pred_file.replace(".json", "_rep.json")
        eval_rep_cmd = ["python", "evaluateRepetition.py", "-s", pred_file,
                        "-r", eval_references[0], "-o", rep_filepath]
        subprocess.call(eval_rep_cmd, cwd=opt.eval_tool_dir)

        metric_filepaths = [lang_file, stat_filepath, rep_filepath]
        all_metrics = merge_dicts([load_json(e) for e in metric_filepaths])
        all_metrics_filepath = pred_file.replace(".json", "_all_metrics.json")
        save_json(all_metrics, all_metrics_filepath, save_pretty=True)

        print("pred_file {} lang_file {}".format(pred_file, lang_file))
        print("[Info] Finished {}.".format(eval_mode))


if __name__ == "__main__":
    main()
