import sys 
import os
import json
import argparse
import csv
import pickle
import copy
import scipy.stats as stats

JSON_COL = "Input.json_variables"
ID= "Input.id"
ANSWER_COL = "Answer.il"

CONTEXT="CONTEXT"
CANDIDATE="CANDIDATE"
CANDIDATES="CANDIDATES"
CAND_ID = "CAND_ID"
SOURCE = "SOURCE"

SOURCE_PMI= "PMI"
SOURCE_CAUSAL="CAUSAL"
SOURCE_LM = "LM"

SHOW = "SHOW"
SAME_AS= "SAME_AS"


def convert_to_readable(csvfile):
    hits = []
    with open(csvfile, 'r') as fi:
        reader = csv.DictReader(fi)
        for row in reader:
            json_cands = json.loads(row[JSON_COL].replace("\\", ""))
            context = json_cands[0]['CONTEXT'] 
            cands = [context]
            for idx, cand in enumerate(json_cands):
                candidate = cand['CANDIDATE']
                score = int(row[ANSWER_COL + str(idx+1)])
                cands.append((candidate, score))
            hits.append(cands)
    return hits
                
def load_vocab(filename):
    #load vocab from json file
    with open(filename, 'rb') as fi:
        voc = pickle.load(fi)
    return voc

def handle_duplicates(row):
    orig_json = json.loads(row[JSON_COL].replace("\\", ""))
    for cand in orig_json[CANDIDATES]:
        if cand[SAME_AS] == -1:
            continue

        dup = [x for x in orig_json[CANDIDATES] if x["POSITION"] == cand[SAME_AS]]
        assert len(dup) == 1
        dup = dup[0]
        duppos = dup["POSITION"]
        row[ANSWER_COL + str(cand["POSITION"])] = row[ANSWER_COL + str(duppos)]
        print(cand[SAME_AS])
    return row

def eval_csv_avg_rank(csv_rows, vocab):

    sys_scores = {SOURCE_PMI: [], SOURCE_CAUSAL: [], SOURCE_LM: []}
    sig_test_scores = {SOURCE_PMI: [], SOURCE_CAUSAL: [], SOURCE_LM: []} #average together the scores for a system for a single row, do sig test on this

    for row in csv_rows:
        row_sys_scores = {SOURCE_PMI: [], SOURCE_CAUSAL: [], SOURCE_LM: []} #scores for the systems in this row (in case multiple candidates per rwo)
        score_positions = []
        row = handle_duplicates(row)
        orig_json = json.loads(row[JSON_COL].replace("\\", ""))
        for cand in orig_json[CANDIDATES]:
            position = cand["POSITION"]
            source = cand[SOURCE]
            score = float(row[ANSWER_COL + str(position)])
            score_positions.append((score, source))
            #sys_scores[source].append(score)
            #row_sys_scores[source].append(score)

        score_positions = sorted(score_positions, key=lambda x: x[0])
        sys_ranks = list(enumerate([x[1] for x in score_positions])) #(rank-1, system name)
        for rank in sys_ranks:
            rank_source = rank[1]
            rank_val = rank[0] + 1
            sys_scores[rank_source].append(rank_val)
            row_sys_scores[rank_source].append(rank_val)

        assert len(row_sys_scores[SOURCE_PMI]) == len(row_sys_scores[SOURCE_CAUSAL]) == len(row_sys_scores[SOURCE_LM])
        sig_test_scores[SOURCE_PMI].append(sum(row_sys_scores[SOURCE_PMI])/len(row_sys_scores[SOURCE_PMI]))
        sig_test_scores[SOURCE_CAUSAL].append(sum(row_sys_scores[SOURCE_CAUSAL])/len(row_sys_scores[SOURCE_CAUSAL]))
        sig_test_scores[SOURCE_LM].append(sum(row_sys_scores[SOURCE_LM])/len(row_sys_scores[SOURCE_LM]))

    assert len(sys_scores[SOURCE_PMI]) == len(sys_scores[SOURCE_CAUSAL]) == len(sys_scores[SOURCE_LM])
    averages = {SOURCE_PMI: sum(sys_scores[SOURCE_PMI])/len(sys_scores[SOURCE_PMI]), 
                SOURCE_CAUSAL: sum(sys_scores[SOURCE_CAUSAL])/len(sys_scores[SOURCE_CAUSAL]), 
                SOURCE_LM: sum(sys_scores[SOURCE_LM])/len(sys_scores[SOURCE_LM])} 
    print(averages)

    wilcox_pmi = stats.wilcoxon(sig_test_scores[SOURCE_CAUSAL], sig_test_scores[SOURCE_PMI])
    wilcox_lm = stats.wilcoxon(sig_test_scores[SOURCE_CAUSAL], sig_test_scores[SOURCE_LM])
    print(wilcox_pmi)
    print(wilcox_lm)

    return averages

def eval_csv_top_count(csv_rows, vocab):

    sys_scores = {SOURCE_PMI: 0, SOURCE_CAUSAL: 0, SOURCE_LM: 0}

    for row in csv_rows:
        score_pos = [] 
        row = handle_duplicates(row)
        orig_json = json.loads(row[JSON_COL].replace("\\", ""))
        for cand in orig_json[CANDIDATES]:
            position = cand["POSITION"]
            source = cand[SOURCE]
            score = float(row[ANSWER_COL + str(position)])
            score_pos.append((score, source))

        score_pos = sorted(score_pos, key=lambda x: x[0], reverse=True)
        top = score_pos[0]
        sys_scores[top[1]] += 1
        
    averages = {SOURCE_PMI: sys_scores[SOURCE_PMI], 
                SOURCE_CAUSAL: sys_scores[SOURCE_CAUSAL], 
                SOURCE_LM: sys_scores[SOURCE_LM]} 
    print(averages)

    return averages

def eval_csv(csv_rows, vocab):

    sys_scores = {SOURCE_PMI: [], SOURCE_CAUSAL: [], SOURCE_LM: []}

    for row in csv_rows:
        row = handle_duplicates(row)
        orig_json = json.loads(row[JSON_COL].replace("\\", ""))
        for cand in orig_json[CANDIDATES]:
            position = cand["POSITION"]
            source = cand[SOURCE]
            score = float(row[ANSWER_COL + str(position)])
            sys_scores[source].append(score)
    assert len(sys_scores[SOURCE_PMI]) == len(sys_scores[SOURCE_CAUSAL]) == len(sys_scores[SOURCE_LM])
    averages = {SOURCE_PMI: sum(sys_scores[SOURCE_PMI])/len(sys_scores[SOURCE_PMI]), 
                SOURCE_CAUSAL: sum(sys_scores[SOURCE_CAUSAL])/len(sys_scores[SOURCE_CAUSAL]), 
                SOURCE_LM: sum(sys_scores[SOURCE_LM])/len(sys_scores[SOURCE_LM])} 
    print(averages)

    return averages

def eval_csv_count_unique(csv_rows, vocab):

    sys_scores = {SOURCE_PMI: [], SOURCE_CAUSAL: [], SOURCE_LM: []}

    for row in csv_rows:
        row = handle_duplicates(row)
        orig_json = json.loads(row[JSON_COL].replace("\\", ""))
        for cand in orig_json[CANDIDATES]:
            position = cand["POSITION"]
            source = cand[SOURCE]
            cand_str = cand[CANDIDATE]
            if '(someone or something)' == cand_str[:22]:
                event_pred = cand_str.split(' ')[3]
                event_pred = event_pred + '->dobj'
            else:
                event_pred = cand_str.split(' ')[1]
                event_pred = event_pred + '->nsubj'

            sys_scores[source].append(event_pred)
    assert len(sys_scores[SOURCE_PMI]) == len(sys_scores[SOURCE_CAUSAL]) == len(sys_scores[SOURCE_LM])
    length = len(sys_scores[SOURCE_PMI])

    averages = {SOURCE_PMI: len(set(sys_scores[SOURCE_PMI])), 
                SOURCE_CAUSAL: len(set(sys_scores[SOURCE_CAUSAL])), 
                SOURCE_LM: len(set(sys_scores[SOURCE_LM]))} 

    averages2 = {SOURCE_PMI: len(set(sys_scores[SOURCE_PMI]))/length, 
                SOURCE_CAUSAL: len(set(sys_scores[SOURCE_CAUSAL]))/length, 
                SOURCE_LM: len(set(sys_scores[SOURCE_LM]))/length} 

    print(averages)
    print(averages2)

    return averages


def eval_csv_and_wilcox(csv_rows, vocab):

    sys_scores = {SOURCE_PMI: [], SOURCE_CAUSAL: [], SOURCE_LM: []}
    sig_test_scores = {SOURCE_PMI: [], SOURCE_CAUSAL: [], SOURCE_LM: []} #average together the scores for a system for a single row, do sig test on this

    for row in csv_rows:
        row_sys_scores = {SOURCE_PMI: [], SOURCE_CAUSAL: [], SOURCE_LM: []}
        row = handle_duplicates(row)
        orig_json = json.loads(row[JSON_COL].replace("\\", ""))
        for cand in orig_json[CANDIDATES]:
            position = cand["POSITION"]
            source = cand[SOURCE]
            score = float(row[ANSWER_COL + str(position)])
            sys_scores[source].append(score)
            row_sys_scores[source].append(score)

        assert len(row_sys_scores[SOURCE_PMI]) == len(row_sys_scores[SOURCE_CAUSAL]) == len(row_sys_scores[SOURCE_LM])
        sig_test_scores[SOURCE_PMI].append(sum(row_sys_scores[SOURCE_PMI])/len(row_sys_scores[SOURCE_PMI]))
        sig_test_scores[SOURCE_CAUSAL].append(sum(row_sys_scores[SOURCE_CAUSAL])/len(row_sys_scores[SOURCE_CAUSAL]))
        sig_test_scores[SOURCE_LM].append(sum(row_sys_scores[SOURCE_LM])/len(row_sys_scores[SOURCE_LM]))



    assert len(sys_scores[SOURCE_PMI]) == len(sys_scores[SOURCE_CAUSAL]) == len(sys_scores[SOURCE_LM])
    averages = {SOURCE_PMI: sum(sys_scores[SOURCE_PMI])/len(sys_scores[SOURCE_PMI]), 
                SOURCE_CAUSAL: sum(sys_scores[SOURCE_CAUSAL])/len(sys_scores[SOURCE_CAUSAL]), 
                SOURCE_LM: sum(sys_scores[SOURCE_LM])/len(sys_scores[SOURCE_LM])} 
    print(averages)

    wilcox_pmi = stats.wilcoxon(sig_test_scores[SOURCE_CAUSAL], sig_test_scores[SOURCE_PMI])
    wilcox_lm = stats.wilcoxon(sig_test_scores[SOURCE_CAUSAL], sig_test_scores[SOURCE_LM])
    print(wilcox_pmi)
    print(wilcox_lm)

    return averages




def aggregate_annotations(csv_rows):
    seen_ids = []
    new_csv_rows = []
    for row in csv_rows:
        if row[ID] not in seen_ids:
            orig_json = json.loads(row[JSON_COL].replace("\\", ""))
            seen_ids.append(row[ID])
            other_rows = [x for x in csv_rows if x[ID] == row[ID] and x is not row]
            newrow = copy.deepcopy(row)
            for idx in range(len(orig_json[CANDIDATES])):
                if row[ANSWER_COL + str(idx)] != "":
                    scores = [float(row[ANSWER_COL + str(idx)])] + [float(x[ANSWER_COL + str(idx)]) for x in other_rows]
                    newscore = sum(scores) / len(scores)
                    newrow[ANSWER_COL + str(idx)] = str(newscore)
            new_csv_rows.append(newrow)
    return new_csv_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Turk')
    parser.add_argument('--turk_csv', type=str, help='File of turk output results' )
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--vocab', type=str, default='data/evocab_freq25')
    parser.add_argument('--aggregate_hits', action='store_true', help='If set, average together multiple annotations of a single hit into a single gold score for the hit, else treat them as seperate')
    
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)

    csv_rows = []
    with open(args.turk_csv, 'r') as fi:
        reader = csv.DictReader(fi)
        data_instances = []
        for row in reader:
            csv_rows.append(row)

    if args.aggregate_hits:
        print("Aggregating Multi-Annotations")
        csv_rows = aggregate_annotations(csv_rows)

    print(len(csv_rows))

 #   instances = eval_csv_and_wilcox(csv_rows, vocab)
 #   instances = eval_csv_avg_rank(csv_rows, vocab)
    #instances = eval_csv_top_count(csv_rows, vocab)
    instances =eval_csv_count_unique(csv_rows, vocab)


