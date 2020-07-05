import json
import numpy as np
import sys


def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f)


def save_json_pretty(data, filepath):
    with open(filepath, "w") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))


def get_ngrams(words_pred, unigrams, bigrams, trigrams, fourgrams):
    # N=1
    for w in words_pred:
        if w not in unigrams:
            unigrams[w] = 0
        unigrams[w] += 1
    # N=2
    for i, w in enumerate(words_pred):
        if i<len(words_pred)-1:
            w_next = words_pred[i+1]
            bigram = '%s_%s' % (w, w_next)
            if bigram not in bigrams:
                bigrams[bigram] = 0
            bigrams[bigram] += 1
    # N=3
    for i, w in enumerate(words_pred):
        if i<len(words_pred)-2:
            w_next = words_pred[i + 1]
            w_next_ = words_pred[i + 2]
            tri = '%s_%s_%s' % (w, w_next, w_next_)
            if tri not in trigrams:
                trigrams[tri] = 0
            trigrams[tri] += 1
    # N=4
    for i, w in enumerate(words_pred):
        if i<len(words_pred)-3:
            w_next = words_pred[i + 1]
            w_next_ = words_pred[i + 2]
            w_next__ = words_pred[i + 3]
            four = '%s_%s_%s_%s' % (w, w_next, w_next_, w_next__)
            if four not in fourgrams:
                fourgrams[four] = 0
            fourgrams[four] += 1
    return unigrams, bigrams, trigrams, fourgrams


def evaluate_repetition(data_predicted, data_gt):
    print('#### Per video ####')

    num_pred = len(data_predicted)
    num_gt = len(data_gt)
    num_evaluated = 0

    re1 = []
    re2 = []
    re3 = []
    re4 = []

    for vid in data_gt:

        unigrams = {}
        bigrams = {}
        trigrams = {}
        fourgrams = {}

        # skip non-existing videos
        if vid not in data_predicted:
            continue

        num_evaluated += 1
        for e in data_predicted[vid]:
            pred_sentence = e["sentence"]

            if pred_sentence[-1] == '.':
                pred_sentence = pred_sentence[0:-1]
            while pred_sentence[-1] == ' ':
                pred_sentence = pred_sentence[0:-1]
            pred_sentence = pred_sentence.replace(',', ' ')
            while '  ' in pred_sentence:
                pred_sentence = pred_sentence.replace('  ', ' ')

            words_pred = pred_sentence.split(' ')
            unigrams, bigrams, trigrams, fourgrams = get_ngrams(words_pred, unigrams, bigrams, trigrams, fourgrams)

        sum_re1 = float(sum([unigrams[f] for f in unigrams]))
        sum_re2 = float(sum([bigrams[f] for f in bigrams]))
        sum_re3 = float(sum([trigrams[f] for f in trigrams]))
        sum_re4 = float(sum([fourgrams[f] for f in fourgrams]))

        vid_re1 = float(sum([max(unigrams[f] - 1, 0) for f in unigrams])) / sum_re1 if sum_re1 != 0 else 0
        vid_re2 = float(sum([max(bigrams[f] - 1, 0) for f in bigrams])) / sum_re2 if sum_re2 != 0 else 0
        vid_re3 = float(sum([max(trigrams[f] - 1, 0) for f in trigrams])) / sum_re3 if sum_re3 != 0 else 0
        vid_re4 = float(sum([max(fourgrams[f]-1, 0) for f in fourgrams])) / sum_re4 if sum_re4 != 0 else 0

        re1.append(vid_re1)
        re2.append(vid_re2)
        re3.append(vid_re3)
        re4.append(vid_re4)

    repetition_scores = dict(
        re1=np.mean(re1),
        re2=np.mean(re2),
        re3=np.mean(re3),
        re4=np.mean(re4),
        num_pred=num_pred,
        num_gt=num_gt,
        num_evaluated=num_evaluated
    )
    return repetition_scores


def evaluate_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--submission", type=str, help="caption submission filepath")
    parser.add_argument("-r", "--reference", type=str, help="GT reference, used to collect the video ids")
    parser.add_argument("-o", "--output", type=str, help="results filepath")
    args = parser.parse_args()

    sub_data = json.load(open(args.submission, "r"))
    ref_data = json.load(open(args.reference, "r"))
    sub_data = sub_data["results"] if "results" in sub_data else sub_data
    ref_data = ref_data["results"] if "results" in ref_data else ref_data
    rep_scores = evaluate_repetition(sub_data, ref_data)

    rep_scores_str = json.dumps(rep_scores, indent=4, sort_keys=True)
    print("Repetition Metrics {}".format(rep_scores_str))

    save_json_pretty(rep_scores, args.output)


if __name__ == '__main__':
    evaluate_main()
