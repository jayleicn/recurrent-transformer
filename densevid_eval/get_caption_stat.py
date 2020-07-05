import json
import argparse
import nltk


def save_json_pretty(data, file_path):
    """save formatted json, use this one for some json config files"""
    with open(file_path, "w") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--submission", type=str, help="submission file")
    parser.add_argument("-v", "--verbose", action="store_true", help="print info")
    parser.add_argument("-r", "--reference", type=str, help="GT reference, used to collect the video ids")
    parser.add_argument("-o", "--output", type=str, help="save path")
    return parser.parse_args()


def get_sen_stat(list_of_str):
    """list_of_str, list(str), str could be a sentence a paragraph"""
    tokenized = [nltk.tokenize.word_tokenize(sen.lower()) for sen in list_of_str]
    num_sen = len(list_of_str)
    lengths = [len(e) for e in tokenized]
    avg_len = 1.0 * sum(lengths) / len(lengths)
    full_vocab = set(flat_list_of_lists(tokenized))
    return {"vocab_size": len(full_vocab), "avg_sen_len": avg_len, "num_sen": num_sen}


def eval_cap():
    """Get vocab size, average length, etc """
    args = get_args()

    # load data
    sub_data = json.load(open(args.submission, "r"))
    ref_data = json.load(open(args.reference, "r"))
    sub_data = sub_data["results"] if "results" in sub_data else sub_data
    ref_data = ref_data["results"] if "results" in ref_data else ref_data
    sub_data = {k: v for k, v in sub_data.items() if k in ref_data}

    submission_data_entries = flat_list_of_lists(sub_data.values())
    submission_sentences = [e["sentence"] for e in submission_data_entries]
    submission_stat = get_sen_stat(submission_sentences)

    if args.verbose:
        for k in submission_stat:
            print("{} submission {}".format(k, submission_stat[k]))
    final_res = {"submission": submission_stat}

    if "gt_sentence" in submission_data_entries[0]:
        gt_sentences = [e["gt_sentence"] for e in submission_data_entries]
        gt_stat = get_sen_stat(gt_sentences)  # only one reference is used here!!!
        final_res["gt_stat"] = gt_stat

    save_json_pretty(final_res, args.output)
    return final_res

if __name__ == '__main__':
    eval_cap()
