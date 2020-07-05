# --------------------------------------------------------
# Dense-Captioning Events in Videos Eval
# Copyright (c) 2017 Ranjay Krishna
# Licensed under The MIT License [see LICENSE for details]
# Written by Ranjay Krishna
# --------------------------------------------------------

import argparse
import string
import json
import sys
sys.path.insert(0, './coco-caption') # Hack to allow the import of pycocoeval

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from sets import Set
import numpy as np

def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

class ANETcaptions(object):
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filenames=None, prediction_filename=None,
                 tious=None, max_proposals=1000,
                 prediction_fields=PREDICTION_FIELDS, verbose=False):
        # Check that the gt and submission files exist and load them
        if len(tious) == 0:
            raise IOError('Please input a valid tIoU.')
        if not ground_truth_filenames:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')

        self.verbose = verbose
        self.tious = tious
        self.max_proposals = max_proposals
        self.pred_fields = prediction_fields
        self.ground_truths = self.import_ground_truths(ground_truth_filenames)
        self.prediction = self.import_prediction(prediction_filename)
        self.tokenizer = PTBTokenizer()

        # Set up scorers, if not verbose, we only use the one we're
        # testing on: METEOR
        if self.verbose:
            self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(),"METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr"),
                (Spice(), "SPICE")
            ]
        else:
            self.scorers = [(Meteor(), "METEOR")]

    def import_prediction(self, prediction_filename):
        if self.verbose:
            print "| Loading submission..."
        submission = json.load(open(prediction_filename))
        if not all([field in submission.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid ground truth file.')
        # Ensure that every video is limited to the correct maximum number of proposals.
        results = {}
        len_captions = 0
        for vid_id in submission['results']:
            results[vid_id] = submission['results'][vid_id][:self.max_proposals]
            len_captions+= len(submission['results'][vid_id][:self.max_proposals])
        print('len of results:', len(results))
        print('len of captions:', len_captions)
        return results

    def import_ground_truths(self, filenames):
        gts = []
        self.n_ref_vids = Set()
        for filename in filenames:
            gt = json.load(open(filename))
            self.n_ref_vids.update(gt.keys())
            gts.append(gt)
        if self.verbose:
            print "| Loading GT. #files: %d, #videos: %d" % (len(filenames), len(self.n_ref_vids))
        return gts

    def iou(self, interval_1, interval_2):
        start_i, end_i = interval_1[0], interval_1[1]
        start, end = interval_2[0], interval_2[1]
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
        iou = float(intersection) / (union + 1e-8)
        return iou

    def check_gt_exists(self, vid_id):
        for gt in self.ground_truths:
            if vid_id in gt:
              return True
        return False

    def get_gt_vid_ids(self):
        vid_ids = set([])
        for gt in self.ground_truths:
            vid_ids |= set(gt.keys())
        return list(vid_ids)

    def evaluate(self):
        aggregator = {}
        self.scores = {}
        for tiou in self.tious:
            scores = self.evaluate_tiou(tiou)
            for metric, score in scores.items():
                if metric not in self.scores:
                    self.scores[metric] = []
                self.scores[metric].append(score)
        if self.verbose:
            self.scores['Recall'] = []
            self.scores['Precision'] = []
            for tiou in self.tious:
                precision, recall = self.evaluate_detection(tiou)
                self.scores['Recall'].append(recall)
                self.scores['Precision'].append(precision)

    def evaluate_detection(self, tiou):
        gt_vid_ids = self.get_gt_vid_ids()
        # Recall is the percentage of ground truth that is covered by the predictions
        # Precision is the percentage of predictions that are valid
        recall = [0] * len(gt_vid_ids)
        precision = [0] * len(gt_vid_ids)
        for vid_i, vid_id in enumerate(gt_vid_ids):
            best_recall = 0
            best_precision = 0
            for gt in self.ground_truths:
                if vid_id not in gt:
                    continue
                refs = gt[vid_id]
                ref_set_covered = set([])
                pred_set_covered = set([])
                num_gt = 0
                num_pred = 0
                if vid_id in self.prediction:
                    for pred_i, pred in enumerate(self.prediction[vid_id]):
                        pred_timestamp = pred['timestamp']
                        for ref_i, ref_timestamp in enumerate(refs['timestamps']):
                            if self.iou(pred_timestamp, ref_timestamp) > tiou:
                                ref_set_covered.add(ref_i)
                                pred_set_covered.add(pred_i)

                    new_precision = float(len(pred_set_covered)) / (pred_i + 1)
                    best_precision = max(best_precision, new_precision)
                new_recall = float(len(ref_set_covered)) / len(refs['timestamps'])
                best_recall = max(best_recall, new_recall)
            recall[vid_i] = best_recall
            precision[vid_i] = best_precision
        return sum(precision) / len(precision), sum(recall) / len(recall)

    def evaluate_tiou(self, tiou):
        # This method averages the tIoU precision from METEOR, Bleu, etc. across videos
        res = {}
        gts = {}
        gt_vid_ids = self.get_gt_vid_ids()

        unique_index = 0

        # video id to unique caption ids mapping
        vid2capid = {}

        cur_res = {}
        cur_gts = {}

        for vid_id in gt_vid_ids:

            vid2capid[vid_id] = []

            # If the video does not have a prediction, then Vwe give it no matches
            # We set it to empty, and use this as a sanity check later on
            if vid_id not in self.prediction:
                pass

            # If we do have a prediction, then we find the scores based on all the
            # valid tIoU overlaps
            else:
                # For each prediction, we look at the tIoU with ground truth
                for i,pred in enumerate(self.prediction[vid_id]):
                    has_added = False
                    for gt in self.ground_truths:
                        if vid_id not in gt:
                            print('skipped')
                            continue
                        gt_captions = gt[vid_id]
                        for caption_idx, caption_timestamp in enumerate(gt_captions['timestamps']):
                            if True or self.iou(pred['timestamp'], caption_timestamp) >= tiou:
                                gt_caption = gt_captions['sentences'][i] # for now we use gt proposal
                                cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                                cur_gts[unique_index] = [{'caption': remove_nonascii(gt_caption)}] # for now we use gt proposal
                                #cur_gts[unique_index] = [{'caption': remove_nonascii(gt_captions['sentences'][caption_idx])}]
                                vid2capid[vid_id].append(unique_index)
                                unique_index += 1
                                has_added = True
                                break # for now we use gt proposal

                            # If the predicted caption does not overlap with any ground truth,
                            # we should compare it with garbage
                            if not has_added:
                                cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                                cur_gts[unique_index] = [{'caption': 'abc123!@#'}]
                                vid2capid[vid_id].append(unique_index)
                                unique_index += 1

        # Each scorer will compute across all videos and take average score
        output = {}

        # call tokenizer here for all predictions and gts
        tokenize_res = self.tokenizer.tokenize(cur_res)
        tokenize_gts = self.tokenizer.tokenize(cur_gts)

        # reshape back
        for vid in vid2capid.keys():
            res[vid] = {index:tokenize_res[index] for index in vid2capid[vid]}
            gts[vid] = {index:tokenize_gts[index] for index in vid2capid[vid]}

        for scorer, method in self.scorers:
            if self.verbose:
                print 'computing %s score...'%(scorer.method())

            # For each video, take all the valid pairs (based from tIoU) and compute the score
            all_scores = {}

            if method == "SPICE": # don't want to compute spice for 10000 times
                print("getting spice score...")
                score, scores = scorer.compute_score(tokenize_gts, tokenize_res)
                all_scores[0] = score
            else:
                for i,vid_id in enumerate(gt_vid_ids):
                    if len(res[vid_id]) == 0 or len(gts[vid_id]) == 0:
                        if type(method) == list:
                            score = [0] * len(method)
                        else:
                            score = 0
                    else:
                        score, scores = scorer.compute_score(gts[vid_id], res[vid_id])
                    all_scores[vid_id] = score

            #print all_scores.values()
            if type(method) == list:
                scores = np.mean(all_scores.values(), axis=0)
                for m in xrange(len(method)):
                    output[method[m]] = scores[m]
                    if self.verbose:
                        print "Calculated tIoU: %1.1f, %s: %0.3f" % (tiou, method[m], output[method[m]])
            else:
                output[method] = np.mean(all_scores.values())
                if self.verbose:
                    print "Calculated tIoU: %1.1f, %s: %0.3f" % (tiou, method, output[method])
        return output

def main(args):
    # Call coco eval
    evaluator = ANETcaptions(ground_truth_filenames=args.references,
                             prediction_filename=args.submission,
                             tious=args.tious,
                             max_proposals=args.max_proposals_per_video,
                             verbose=args.verbose)
    evaluator.evaluate()

    # Output the results
    if args.verbose:
        for i, tiou in enumerate(args.tious):
            print '-' * 80
            print "tIoU: " , tiou
            print '-' * 80
            for metric in evaluator.scores:
                score = evaluator.scores[metric][i]
                print '| %s: %2.4f'%(metric, 100*score)

    # Print the averages
    print '-' * 80
    print "Average across all tIoUs"
    print '-' * 80
    output = {}
    for metric in evaluator.scores:
        score = evaluator.scores[metric]
        print '| %s: %2.4f'%(metric, 100 * sum(score) / float(len(score)))
    output[metric] = 100 * sum(score) / float(len(score))
    json.dump(output,open(args.output,'w'))
    print(output)
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate the results stored in a submissions file.')
    parser.add_argument('-s', '--submission', type=str,  default='sample_submission.json',
                        help='sample submission file for ActivityNet Captions Challenge.')
    parser.add_argument('-r', '--references', type=str, nargs='+', default=['data/val_1.json'],
                        help='reference files with ground truth captions to compare results against. delimited (,) str')
    parser.add_argument('-o', '--output', type=str,  default='result.json',
                        help='output file with final language metrics.')
    parser.add_argument('--tious', type=float,  nargs='+', default=[0.3],
                        help='Choose the tIoUs to average over.')
    parser.add_argument('-ppv', '--max-proposals-per-video', type=int, default=1000,
                        help='maximum propoasls per video.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print intermediate steps.')
    args = parser.parse_args()

    main(args)
