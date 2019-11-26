import sys 
import os
import json
import argparse
import csv
import pickle

ACCEPTABLE_OUT = [3,4]  #Which scores do we include in the data, default is 3 (more likely) and 4 (almost certain)
JSON_COL = "Input.json_variables"
ANSWER_COL = "Answer.statement_likelihood_"

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

def recreate_row(row):
    newrow = {}
    newrow['e1'] = row['e1']
    newrow['e2'] = row['e2']
    newrow['e1_text'] = row['e1_text']
    newrow['id'] = row['id']
    newrow['e1prev_intext'] = row['e1prev_intext']
    return newrow

def check_rows(row):
    cands = []
    for cand in row:
        cands.append(recreate_row(cand))
    assert all([x == cands[0] for x in cands])

def process_csv(csvfile, vocab):

    with open(csvfile, 'r') as fi:
        reader = csv.DictReader(fi)
        data_instances = []
        for row in reader:
            json_cands = json.loads(row[JSON_COL].replace("\\", ""))
            check_rows(json_cands)
            newrow = recreate_row(json_cands[0]) 
            
            e1prev_outtext = []
            for idx, cand in enumerate(json_cands):
                candidate = cand['CANDIDATE_EVENT_FORM']
                score = int(row[ANSWER_COL + str(idx + 1)])
                if score in ACCEPTABLE_OUT:
                    if candidate in vocab.stoi: 
                        e1prev_outtext.append(candidate)
                    else:
                        print("{} is not in vocab, skipping...".format(candidate))

            if e1prev_outtext:
                 print("Creating New Row with {} Out of Text Entries".format(len(e1prev_outtext)))
            else:
                print("No valid Out of Text Events")

            newrow['e1prev_outtext'] = e1prev_outtext
            data_instances.append(newrow)

    return data_instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Turk')
    parser.add_argument('--turk_csv', type=str, help='File of turk output results' )
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--vocab', type=str, default='data/evocab_freq25')
    
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)

    instances = process_csv(args.turk_csv, vocab)

    with open(args.outfile, 'w') as fi:
        for inst in instances:
            json.dump(inst, fi)
            fi.write("\n")
            
