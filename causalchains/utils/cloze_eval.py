import argparse
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from causalchains.utils.data_utils import EOS_TOK, SOS_TOK
import causalchains.utils.data_utils as du
import json
import csv
import pickle
import random
import logging
import math

def top_causal_choices(chain, scores, evocab, so_events):
    """
    return back a list of the top choices according to the causal model
    for a ending to the chain
    params:
        (str) chain : a list of string representation of the event
        scores : the output of causalchains.train.testing.normalized_intervention_mat, a tuple
                 with (interven_dists, so_events_itos, so_events_stoi), where intervention dist is a matrix
                 whose jth column is a normalized selection of potential causes of e2
        evocab: The original event vocabulary
    """
    score_mat = torch.Tensor(scores[0])
    evocab_scores = []
    #for idx, last_e in enumerate(evocab.itos):
    for idx, last_e in enumerate(so_events):
        #if last_e in so_events:
        scor = 0
        prev_event_scores = score_mat[:, evocab.stoi[last_e]].tolist() #compatibility scores for previous events for this vocab item
        for ev in chain:
            if ev in scores[2]:
                scor += prev_event_scores[scores[2][ev]]
            else:
                scor += 0
        scor = scor / len(chain)
        evocab_scores.append((last_e, scor))
            
    evocab_scores = sorted(evocab_scores, key=lambda x: x[1], reverse=True)     
        

    return evocab_scores

def top_pmi_choices(chain, scores, evocab, so_events):
    """
    return back a list of the top choices according to the causal model
    for a ending to the chain
    params:
        (str) chain : a list of string representation of the event
        scores :pmi_dict, map event to list of ([prev_e, pmi], [prev_e, pmi],...) list of tuples of top previous pmi pairs
        evocab: The original event vocabulary

    """
    evocab_scores = []
#    for idx, last_e in enumerate(evocab.itos):
    for idx, last_e in enumerate(so_events):
        #if last_e in so_events:
        scor = 0
        prev_event_scores = dict(scores[last_e]) #dict mapping events to their forward pmi with last_e
        for ev in chain:
            if ev in prev_event_scores:
                scor += prev_event_scores[ev]
            else: 
                scor += 0
        scor = scor / len(chain)
        evocab_scores.append((last_e, scor))
            
    evocab_scores = sorted(evocab_scores, key=lambda x: x[1], reverse=True)     

    return evocab_scores

def top_lm_choices(args, model, example, evocab, so_events):
    """
    Complete generation of an event chain given prefix example
    params:
        causalchains.models.LM (model)
        example:  List[seqlens] - List of the (string form) of events, to be converted to readable inputs here
    """

    model.eval()
    outputs = list(example)
    evocab_scores = []

    #Process the prefix
    text_inst= torch.LongTensor([evocab.stoi[SOS_TOK]] + [evocab.stoi[x] for x in example]).to(device=args.device) #seqlen tensor]
    
    hidden=None
    logits = None
    for step in range(text_inst.shape[0]):
        step_inp = text_inst[step] #get all instances for this step
        step_inp = step_inp.unsqueeze(0).unsqueeze(0) #[1 X 1]

        logit_i, hidden = model(step_inp, hidden)
        logits = logit_i #[1 X vocab]

    logits = logits.squeeze(dim=0).cpu().tolist()
    for idx, score in enumerate(logits):
        if evocab.itos[idx] in so_events:
            evocab_scores.append((evocab.itos[idx], score))


    evocab_scores = sorted(evocab_scores, key=lambda x: x[1], reverse=True)     

    return evocab_scores


def cloze_eval(args, cloze_data, pmi_dict, causal_dict, evocab, so_events, recall_at, threshold, evocab_lm=None, lm_model=None):
    pmi_res = []
    causal_res = []
    lm_res = []

    for idx, instance in enumerate(cloze_data):
        chain = instance[0]
        answer = instance[1]
        top_causal = [x[0] for x in top_causal_choices(chain, causal_dict, evocab, so_events)][:recall_at]
        top_pmi = [x[0] for x in top_pmi_choices(chain, pmi_dict, evocab, so_events)][:recall_at]
        top_lm = [x[0] for x in top_lm_choices(args, lm_model, chain, evocab_lm, so_events)][:recall_at]

        if answer in top_causal:
            causal_res.append(1)
#            print("CHAIN {}, ANS {}".format(chain, answer))
        else:
            causal_res.append(0)
#            print("Wrong CHAIN {}, ANS {}".format(chain, answer))

        if answer in top_pmi:
            pmi_res.append(1)
        else:
            pmi_res.append(0)

        if answer in top_lm:
            lm_res.append(1)
        else:
            lm_res.append(0)


        print("Eval Line {}, Threshold {}, Recall at {}, Causal {}, LM {}, PMI {}".format(idx, threshold, recall_at, answer in top_causal, answer in top_lm , answer in top_pmi))

        if idx % 25 == 0 and idx != 0:
            length = len(causal_res)
            print("\nCurrent EVAL, RECALL AT {}\n".format(recall_at))
            print("Causal Recall at {}: {}".format(recall_at, sum(causal_res) / length))
            print("PMI Recall at {}: {}".format(recall_at, sum(pmi_res) / length))
            print("LM Recall at {}: {}".format(recall_at, sum(lm_res) / length))


    assert len(causal_res) == len(pmi_res) == len(lm_res)
    length = len(causal_res)

    print("\nFINAL EVAL, RECALL AT {}\n".format(recall_at))
    print("Causal Recall at {}: {}".format(recall_at, sum(causal_res) / length))
    print("PMI Recall at {}: {}".format(recall_at, sum(pmi_res) / length))
    print("LM Recall at {}: {}".format(recall_at, sum(lm_res) / length))

def cloze_eval_causal(args, cloze_data, pmi_dict, causal_dict, evocab, so_events, recall_at, threshold, evocab_lm=None, lm_model=None):
    pmi_res = []
    causal_res = []
    lm_res = []

    for idx, instance in enumerate(cloze_data):
        chain = instance[0]
        answer = instance[1]
        top_causal = [x[0] for x in top_causal_choices(chain, causal_dict, evocab, so_events)][:recall_at]

        ####

#        top_causal2 = [x[0] for x in top_causal_choices(chain, causal_dict, evocab, so_events)]
#        top_causal_scores = top_causal_choices(chain, causal_dict, evocab, so_events)
#        ans_ind = top_causal2.index(answer)
#
#        
#        scores_mat = torch.Tensor(causal_dict[0])
#        prev_event_scores = scores_mat[:, evocab.stoi[answer]].tolist() #compatibility scores for previous events for this vocab item
#        ans_foo = []
#        for ev in chain:
#            if ev in causal_dict[2]:
#                ans_foo.append(math.log(prev_event_scores[causal_dict[2][ev]]))
#            else:
#                ans_foo.append(0)
#
       # print("Chain: {}".format(chain))
       # print("Ans: {}".format(ans_foo))
       # prev_event_scores = scores_mat[:, evocab.stoi[top_causal[149]]].tolist() #compatibility scores for previous events for this vocab item
       # top_foo = []
       # for ev in chain:
       #     if ev in causal_dict[2]:
       #         top_foo.append(math.log(prev_event_scores[causal_dict[2][ev]]))
       #     else:
       #         top_foo.append(0)

       # print("Top: {}".format(top_foo))
        ####

        if answer in top_causal:
            causal_res.append(1)
            print("CHAIN {}, ANS {}".format(chain, answer))
        else:
            causal_res.append(0)


        print("Eval Line {}, Threshold {}, Recall at {}, Causal {}".format(idx, threshold, recall_at, answer in top_causal))

        if idx % 25 == 0 and idx != 0:
            length = len(causal_res)
            print("\nCurrent EVAL, RECALL AT {}\n".format(recall_at))
            print("Causal Recall at {}: {}".format(recall_at, sum(causal_res) / length))


    length = len(causal_res)

    print("\nFINAL EVAL, RECALL AT {}\n".format(recall_at))
    print("Causal Recall at {}: {}".format(recall_at, sum(causal_res) / length))


def cloze_eval_ranking(args, cloze_data, pmi_dict, causal_dict, evocab, so_events, recall_at, evocab_lm=None, lm_model=None):
    pmi_res = []
    causal_res = []
    lm_res = []

    for idx, instance in enumerate(cloze_data):
        chain = instance[0]
        answer = instance[1]
        top_causal = [x[0] for x in top_causal_choices(chain, causal_dict, evocab, so_events)]
        top_pmi = [x[0] for x in top_pmi_choices(chain, pmi_dict, evocab, so_events)]
        top_lm = [x[0] for x in top_lm_choices(args, lm_model, chain, evocab_lm, so_events)]

        if len(top_causal) == len(top_pmi) == len(top_lm):
            pmi_res.append(top_pmi.index(answer))
            causal_res.append(top_causal.index(answer))
            lm_res.append(top_lm.index(answer))
            print("Eval Line {}, Ranking, Causal {}, LM {}, PMI {}".format(idx, causal_res[-1], lm_res[-1], pmi_res[-1]))
        else:
            print("Unequal top lists, skipping")

    assert len(causal_res) == len(pmi_res) == len(lm_res)
    length = len(causal_res)

    print("\nFINAL EVAL, RECALL AT {}\n".format(recall_at))
    print("Causal Avg Rank: {}".format(sum(causal_res) / length))
    print("PMI Avg Rank: {}".format(sum(pmi_res) / length))
    print("LM Avg Rank: {}".format(sum(lm_res) / length))


def load_cloze_data(txt_file, evocab, so_events, threshold):
    #Return back list of tuples ([chain], ans)
    cloze_data = []
    with open(txt_file, 'r') as fi:
        for line in fi:
            chain = line.split("<ANSWER>")[0].strip().split(" ")
            ans= line.split("<ANSWER>")[1].strip()
            if ans not in so_events[:threshold]: #only include instances where the answer is not in top k most freq
                cloze_data.append((chain, ans))

    return cloze_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CreateCandidates')
    parser.add_argument('--pmi_dict', type=str, help='pmi json information')
    parser.add_argument('--evocab', type=str, help='the event vocabulary pickle file', default='./data/evocab_freq25')
    parser.add_argument('--causal_dict', type=str, help='Matrix output of causal model, output of causalchains.train.testing.normalized_score_matrix')
    parser.add_argument('--cloze_data', type=str)
    parser.add_argument('--recall_at', type=int, default=50, help="Recall@_")
    parser.add_argument('--lm_model', type=str, default=None)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--ranking', action='store_true')
    parser.add_argument('--threshold', type=int, default=100, help="Don't count vocab items in top k list")
    args = parser.parse_args()

    args.device=None

    if torch.cuda.is_available():
        if not args.cuda:
            logging.warning("WARNING: You have a CUDA device, so you should probably run with --cuda")
            args.device = torch.device('cpu')
        else:
            args.device = torch.device('cuda')

            logging.info("Using GPU {}".format(torch.cuda.get_device_name(args.device)))

    else:
        args.device = torch.device('cpu')
    


    evocab = du.load_vocab(args.evocab)


    with open(args.pmi_dict, 'r') as fi:
        pmi_dict = json.load(fi)

    with open(args.causal_dict, 'rb') as fi:
        causal_dict = pickle.load(fi)

    evocab_lm = du.convert_to_lm_vocab(copy.deepcopy(evocab))
    lm_model = torch.load(args.lm_model, map_location=args.device)

    
    so_events = [x for x in evocab.itos if len(x.split('->')) == 2 and x.split('->')[1] in ['nsubj', 'dobj', 'iobj']] #only count these in the rankings

    print(len(so_events))

    cloze_data = load_cloze_data(args.cloze_data, evocab, so_events, args.threshold)
    print(len(cloze_data))


    cloze_data = cloze_data[:1000]

    if args.ranking:
        cloze_eval_ranking(args, cloze_data, pmi_dict, causal_dict, evocab, so_events, args.recall_at, evocab_lm, lm_model)
    else:
        cloze_eval(args, cloze_data, pmi_dict, causal_dict, evocab, so_events, args.recall_at, args.threshold, evocab_lm, lm_model)
               
            


        



