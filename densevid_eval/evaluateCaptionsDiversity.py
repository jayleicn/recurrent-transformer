import json
import numpy as np
import sys

def getNgrams(words_pred, unigrams, bigrams, trigrams, fourgrams):
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

def evaluateDiversity(input_file):
    dp = json.load(open(input_file, 'r'))
    dg = json.load(open('anet_data/video_gt.json', 'r'))
    da = json.load(open('anet_data/activity_net.v1-3.min.json', 'r'))

    overall_stats(dp, dg)
    activity_stats(dp, dg, da)
    video_stats(dp, dg)

def overall_stats(data_predicted, data_gt):
    print('#### Overall ####')

    of = open('anet_data/train.json')
    queries = json.load(of)
    trainingQueries = {}  # all training sentences

    for q in queries:
        qs = queries[q]['sentences']
        for query in qs:
            if query == "": break
            query = query.lower()
            # clean punctuation
            query = query.replace(',', ' ')
            query = query.replace('.', ' ')
            query = query.replace(':', ' ')
            query = query.replace(';', ' ')
            query = query.replace('!', ' ')
            query = query.replace('?', ' ')
            query = query.replace('"', ' ')
            query = query.replace('@', ' ')
            query = query.replace('(', ' ')
            query = query.replace(')', ' ')
            query = query.replace('[', ' ')
            query = query.replace(']', ' ')
            query = query.replace('<', ' ')
            query = query.replace('>', ' ')
            query = query.replace('`', ' ')
            query = query.replace('#', ' ')
            query = query.replace(u'\u2019', "'")
            while query[-1] == ' ':
                query = query[0:-1]
            while query[0] == ' ':
                query = query[1:]
            while '  ' in query:
                query = query.replace('  ', ' ')
            # print(query)
            if query not in trainingQueries:
                trainingQueries[query] = 0
            trainingQueries[query] += 1

    vocab = {}
    novel_sentence = []
    uniq_sentence = {}
    count_sent = 0
    sent_length = []

    for vid in data_gt['results']:
        for i, _ in enumerate(data_gt['results'][vid]):

            try:
                pred_sentence = data_predicted['results'][vid][i]['sentence'].lower()
            except:
                continue

            if pred_sentence[-1] == '.':
                pred_sentence = pred_sentence[0:-1]
            while pred_sentence[-1] == ' ':
                pred_sentence = pred_sentence[0:-1]
            pred_sentence = pred_sentence.replace(',', ' ')
            while '  ' in pred_sentence:
                pred_sentence = pred_sentence.replace('  ', ' ')

            if pred_sentence in trainingQueries:
                novel_sentence.append(0)
            else:
                novel_sentence.append(1)

            if pred_sentence not in uniq_sentence:
                uniq_sentence[pred_sentence] = 0
            uniq_sentence[pred_sentence] += 1

            words_pred = pred_sentence.split(' ')
            for w in words_pred:
                if w not in vocab:
                    vocab[w] = 0
                vocab[w] += 1

            sent_length.append(len(words_pred))
            count_sent += 1

    print ('Vocab: %d\t Novel Sent: %.2f\t Uniq Sent: %.2f\t Sent length: %.2f' %
           (len(vocab), np.mean(novel_sentence), len(uniq_sentence)/float(count_sent), np.mean(sent_length)))

def activity_stats(data_predicted, data_gt, data_annos):
    print('#### Per activity ####')

    # Per activity
    data_annos = data_annos['database']
    activities = {}
    vid2act = {}
    for vid_id in data_annos:
        vid2act[vid_id] = []
        annos = data_annos[vid_id]['annotations']
        for anno in annos:
            if anno['label'] not in vid2act[vid_id]:
                vid2act[vid_id].append(anno['label'])
                if anno['label'] not in activities:
                    activities[anno['label']] = True
    div1 = {}
    div2 = {}
    div3 = {}
    div4 = {}
    re = {}
    sentences = {}

    for act in activities:
        div1[act] = -1
        div2[act] = -1
        div3[act] = -1
        div4[act] = -1
        re[act] = -1
        sentences[act] = []

    for vid in data_gt['results']:

        act = vid2act[vid[2:]][0]

        if vid not in data_predicted['results']:
            continue

        for i, _ in enumerate(data_gt['results'][vid]):

            try:
                pred_sentence = data_predicted['results'][vid][i]['sentence']
            except:
                continue

            if pred_sentence[-1] == '.':
                pred_sentence = pred_sentence[0:-1]
            while pred_sentence[-1] == ' ':
                pred_sentence = pred_sentence[0:-1]
            pred_sentence = pred_sentence.replace(',', ' ')
            while '  ' in pred_sentence:
                pred_sentence = pred_sentence.replace('  ', ' ')

            sentences[act].append(pred_sentence)

    for act in activities:
        unigrams = {}
        bigrams = {}
        trigrams = {}
        fourgrams = {}

        for pred_sentence in sentences[act]:
            words_pred = pred_sentence.split(' ')
            unigrams, bigrams, trigrams, fourgrams = getNgrams(words_pred, unigrams, bigrams, trigrams, fourgrams)

        sum_unigrams = sum([unigrams[un] for un in unigrams])
        vid_div1 = float(len(unigrams)) / float(sum_unigrams)
        vid_div2 = float(len(bigrams)) / float(sum_unigrams)
        vid_div3 = float(len(trigrams)) / float(sum_unigrams)
        vid_div4 = float(len(fourgrams)) / float(sum_unigrams)

        vid_re = float(sum([max(fourgrams[f]-1,0) for f in fourgrams])) / float(sum([fourgrams[f] for f in fourgrams]))

        div1[act] = vid_div1
        div2[act] = vid_div2
        div3[act] = vid_div3
        div4[act] = vid_div4
        re[act] = vid_re

    mean_div1 = np.mean([div1[act] for act in activities])
    mean_div2 = np.mean([div2[act] for act in activities])
    mean_div3 = np.mean([div3[act] for act in activities])
    mean_div4 = np.mean([div4[act] for act in activities])
    mean_re = np.mean([re[act] for act in activities])

    print ('Div-1: %.4f\t Div-2: %.4f\t RE: %.4f' % (mean_div1, mean_div2, mean_re))

def video_stats(data_predicted, data_gt):
    print('#### Per video ####')

    # Per video

    div1 = []
    div2 = []
    div3 = []
    div4 = []
    re1 = []
    re2 = []
    re3 = []
    re4 = []

    for vid in data_gt['results']:

        unigrams = {}
        bigrams = {}
        trigrams = {}
        fourgrams = {}

        if vid not in data_predicted['results']:
            continue

        for i, _ in enumerate(data_gt['results'][vid]):

            try:
                pred_sentence = data_predicted['results'][vid][i]['sentence']
            except:
                continue

            if pred_sentence[-1] == '.':
                pred_sentence = pred_sentence[0:-1]
            while pred_sentence[-1] == ' ':
                pred_sentence = pred_sentence[0:-1]
            pred_sentence = pred_sentence.replace(',', ' ')
            while '  ' in pred_sentence:
                pred_sentence = pred_sentence.replace('  ', ' ')

            words_pred = pred_sentence.split(' ')
            unigrams, bigrams, trigrams, fourgrams = getNgrams(words_pred, unigrams, bigrams, trigrams, fourgrams)

        sum_unigrams = sum([unigrams[un] for un in unigrams])
        vid_div1 = float(len(unigrams)) / float(sum_unigrams)
        vid_div2 = float(len(bigrams)) / float(sum_unigrams)
        vid_div3 = float(len(trigrams)) / float(sum_unigrams)
        vid_div4 = float(len(fourgrams)) / float(sum_unigrams)

        vid_re1 = float(sum([max(unigrams[f] - 1, 0) for f in unigrams])) / float(sum([unigrams[f] for f in unigrams]))
        vid_re2 = float(sum([max(bigrams[f] - 1, 0) for f in bigrams])) / float(sum([bigrams[f] for f in bigrams]))
        vid_re3 = float(sum([max(trigrams[f] - 1, 0) for f in trigrams])) / float(sum([trigrams[f] for f in trigrams]))
        vid_re4 = float(sum([max(fourgrams[f]-1,0) for f in fourgrams])) / float(sum([fourgrams[f] for f in fourgrams]))

        div1.append(vid_div1)
        div2.append(vid_div2)
        div3.append(vid_div3)
        div4.append(vid_div4)
        re1.append(vid_re1)
        re2.append(vid_re2)
        re3.append(vid_re3)
        re4.append(vid_re4)

    print ('tDiv-1: %.4f\t Div-2: %.4f\t RE-4: %.4f' % (
        np.mean(div1), np.mean(div2),np.mean(re4)))

if __name__=='__main__':
    submission = sys.argv[1]
    evaluateDiversity(submission)
