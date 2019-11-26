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
from pattern.en import conjugate

BAD_WORDS = []

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

def top_causal_choices(chain, scores, evocab):
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
    for idx, last_e in enumerate(evocab.itos):
        if all([usable(e2, last_e) for e2 in chain]):
            scor = 0
            prev_event_scores = score_mat[:, evocab.stoi[last_e]].tolist() #compatibility scores for previous events for this vocab item
            for ev in chain:
                scor += prev_event_scores[scores[2][ev]]
            scor = scor / len(chain)
            evocab_scores.append((last_e, scor))
            
    evocab_scores = sorted(evocab_scores, key=lambda x: x[1], reverse=True)     
        

    return evocab_scores

def top_pmi_choices(chain, scores, evocab):
    """
    return back a list of the top choices according to the causal model
    for a ending to the chain
    params:
        (str) chain : a list of string representation of the event
        scores :pmi_dict, map event to list of ([prev_e, pmi], [prev_e, pmi],...) list of tuples of top previous pmi pairs
        evocab: The original event vocabulary

    """
    evocab_scores = []
    for idx, last_e in enumerate(evocab.itos):
        if all([usable(e2, last_e) for e2 in chain]):
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

def usable(e2, e1):
    """
    Return true is e1 is a usuable output for e2, use this for all models 

    Removes repeats and events without nsubj dobj or iobj
    """

    #Various 'mispelled' words that might be confusing for annotation, just remove them
    nongrammatical = ['thinkin', 'talkin', 'sayin', 'doin', 'lookin', 'gettin', 'tellin', 'fuckin', 'askin', 'walkin', 'makin', 'feelin', 'leavin', 'comin', 'sittin', 'stayin', 'wonderin', 'seein', 'drivin', 'playin']
    modals = ['didnt', 'dont', 'wouldnt', 'couldnt', 'shouldnt', 'didna', 'hadnt', 'doesnt', 'wasnt', 'canna']

    if not len(e1.split('->')) == 2:
        return False
    if e2 == e1:
        return False
    if e1.split('->')[0] == e2.split('->')[0]:
        return False
    if e1.split('->')[1] not in ['nsubj', 'dobj', 'iobj']:
        return False
    if e1.split('->')[0] in nongrammatical: #these colliqual expresions (ie talkin) can be hard for turkers to annotate
        return False
    if e1.split('->')[0] in modals: #dont include contracted negation events 
        return False
    if '*' in e1.split('->')[0] or "'" in e1.split('->')[0] or '"' in e1.split('->')[0] or '-' in e1.split('->')[0]: 
        return False
    if len(e1.split('->')[0]) < 3:
        return False
    if e1.split('->')[0] in BAD_WORDS:
        return False

    return True

def convert_to_text(cand_pred, e1_arg, conj_pred=True):
    event = cand_pred
    e1_rel = cand_pred.split('->')[1]
    cand_pred = cand_pred.split('->')[0]

    if conj_pred:
        if e1_arg.lower() == 'you':
            conj = conjugate(cand_pred, '2sgp')
        else:
            conj = conjugate(cand_pred, '1sgp')
        text_cand_pred = conj if conj else cand_pred
    else:
        text_cand_pred = cand_pred

    if e1_rel == 'nsubj':
        event_text = e1_arg + " " + text_cand_pred + " " + "(something or someone)"
    elif e1_rel == 'iobj':
        event_text = "(Someone or something)" + " " + text_cand_pred + " " + "something to " + e1_arg.capitalize()
    else:
        event_text = "(Someone or something)" + " " + text_cand_pred + " " + e1_arg.capitalize()

    return event_text.capitalize()


def top_lm_choices(model, example, evocab):
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
    text_inst= torch.LongTensor([evocab.stoi[SOS_TOK]] + [evocab.stoi[x] for x in example]) #seqlen tensor]
    
    hidden=None
    logits = None
    for step in range(text_inst.shape[0]):
        step_inp = text_inst[step] #get all instances for this step
        step_inp = step_inp.unsqueeze(0).unsqueeze(0) #[1 X 1]

        logit_i, hidden = model(step_inp, hidden)
        logits = logit_i #[1 X vocab]

    #decode
    already_used = [evocab.stoi[x] for x in outputs] + [evocab.stoi['<unk>']]
    for idx in already_used:
        logits[0, idx] = -1e10

    logits = logits.squeeze(dim=0).tolist()
    for idx, score in enumerate(logits):
        if all([usable(e2, evocab.itos[idx]) for e2 in example]):
            evocab_scores.append((evocab.itos[idx], score))


    evocab_scores = sorted(evocab_scores, key=lambda x: x[1], reverse=True)     

    return evocab_scores


def full_chains(chain_list, pmi_dict, causal_dict, evocab, evocab_lm=None, lm_model=None):
    """
    Return back the top candidates for e4 for each chain in chain_list (a list of three events in string form) 
    params:
        lists of tuples (e2, [list of events in str form])
        
    """
    pmi_res = []
    causal_res = []
    lm_res = []

    for idx, chain in enumerate(chain_list):
        top_causal = [x[0] for x in top_causal_choices(chain, causal_dict, evocab) if x[0] not in evocab.itos[:22]]
        top_pmi = [x[0] for x in top_pmi_choices(chain, pmi_dict, evocab) if x[0] not in evocab.itos[:22]]
        if evocab_lm is not None:
            top_lm = [x[0] for x in top_lm_choices(lm_model, chain, evocab_lm) if x[0] not in evocab.itos[:22]]

        causal_res.append((chain, top_causal[:1]))
        pmi_res.append((chain, top_pmi[:1]))
        lm_res.append((chain, top_lm[:1]))
        print("Causal Processed event {}: {}".format(chain,top_causal[:1]))
        print("PMI Processed event {}: {}".format(chain,top_pmi[:1]))
        print("LM Processed event {}: {}".format(chain,top_lm[:1]))
        print("-------------------------------------------------")

    return (pmi_res, causal_res, lm_res)


def process_chain_candidates(txt_file, evocab):
    chain_cands = []
    with open(txt_file, 'r') as fi:
        for line in fi:
            events = line.strip().split(" ")
            if len(events) == 3:
                e1 = events[0] + '->nsubj'
                e2 = events[1] + '->nsubj'
                e3 = events[2] + '->nsubj'
                if e1 in evocab.itos[:2500] and e2 in evocab.itos[:2500] and e3 in evocab.itos[:2500] and len(set([e1,e2,e3]))==3:
                    chain_cands.append([e1, e2, e3])
    return chain_cands
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CreateCandidates')
    parser.add_argument('--pmi_dict', type=str, help='pmi json information')
    parser.add_argument('--evocab', type=str, help='the event vocabulary pickle file', default='./data/evocab_freq25')
    parser.add_argument('--causal_dict', type=str, help='Matrix output of causal model, output of causalchains.train.testing.normalized_score_matrix')
    parser.add_argument('--bad_words', type=str, default='data/bad-words.txt')
    parser.add_argument('--candidates', type=str, default='data/chain_candidates.txt')
    parser.add_argument('--turk_format', action='store_true')
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--lm_model', type=str, default=None)
    args = parser.parse_args()


    if args.turk_format:
        outwriter = open(args.outfile, 'w')
        csvwriter= csv.writer(outwriter)
        csvwriter.writerow(['id', 'json_variables'])


    evocab = du.load_vocab(args.evocab)


    with open(args.pmi_dict, 'r') as fi:
        pmi_dict = json.load(fi)

    with open(args.causal_dict, 'rb') as fi:
        causal_dict = pickle.load(fi)

    global BAD_WORDS

    with open(args.bad_words, 'r') as fi:
        bword_lines = fi.readlines()
        BAD_WORDS = [x.rstrip() for x in bword_lines]


    if args.lm_model is not None:
        evocab_lm = du.convert_to_lm_vocab(copy.deepcopy(evocab))
        lm_model = torch.load(args.lm_model, map_location=torch.device('cpu'))
    else:
        evocab_lm = None
        lm_model= None


    chain_cands = process_chain_candidates(args.candidates, evocab)

    pmi_res, causal_res, lm_res = full_chains(chain_cands, pmi_dict, causal_dict, evocab, evocab_lm, lm_model)


    if not args.turk_format:
        with open(args.outfile, 'w') as fi:
            if args.lm_model is not None:
                for idx, (pmi, causal, lm) in enumerate(zip(pmi_res, causal_res, lm_res)):
                    assert pmi[0] == causal[0] == lm[0]
                    fi.write("Chain2: {}\n".format(pmi[0]))
                    fi.write("\tPMI choices for Event 1: {}\n".format(",".join(pmi[1])))
                    fi.write("\tCausal choices for Event 1: {}\n".format(",".join(causal[1])))
                    fi.write("\tLm choices for Event 1: {}\n".format(",".join(lm[1])))
                    fi.write("----------------------------------------------------\n")

            else:
                for idx, (pmi, causal) in enumerate(zip(pmi_res, causal_res)):
                    assert pmi[0] == causal[0]
                    fi.write("Event 2: {}\n".format(pmi[0]))
                    fi.write("\tPMI choices for Event 1: {}\n".format(",".join(pmi[1])))
                    fi.write("\tCausal choices for Event 1: {}\n".format(",".join(causal[1])))
                    fi.write("----------------------------------------------------\n")
    else:
        names = ["Steve", "Sarah", "Trish", "Ivan", "Dmitri", "Dorthy"]
        for idx, (pmi, causal, lm) in enumerate(zip(pmi_res, causal_res, lm_res)):
            assert pmi[0] == causal[0] == lm[0]
            assert len(pmi[0]) == 3
            protag = random.choice(names)
            hitjson = {}

            hitjson[CONTEXT + "1"] = convert_to_text(pmi[0][0], protag, conj_pred=True)
            hitjson[CONTEXT + "2"] = convert_to_text(pmi[0][1], protag, conj_pred=True)
            hitjson[CONTEXT + "3"] = convert_to_text(pmi[0][2], protag, conj_pred=True)

            cand_questions = []

            if len(pmi[1]) != 1 or len(causal[1]) != 1 or len(lm[1]) != 1:
                print("Skipping candidate {}, less than two choices for one source".format(pmi[0]))
                continue

            for event in pmi[1]:
                cand_questions.append({CANDIDATE: convert_to_text(event, protag, conj_pred=True), SOURCE: SOURCE_PMI})
            for event in causal[1]:
                cand_questions.append({CANDIDATE: convert_to_text(event, protag, conj_pred=True), SOURCE: SOURCE_CAUSAL})
            for event in lm[1]:
                cand_questions.append({CANDIDATE: convert_to_text(event, protag, conj_pred=True), SOURCE: SOURCE_LM})


            random.shuffle(cand_questions)
            for position, ques in enumerate(cand_questions):
                ques["POSITION"] = position

            #Take care of duplicate candidates
            seen = {} 
            for ques in cand_questions:
                if ques[CANDIDATE] not in seen.keys():
                    seen[ques[CANDIDATE]] = ques["POSITION"]
                    ques[SHOW]=1
                    ques[SAME_AS]=-1
                else:
                    ques[SHOW]=0
                    ques[SAME_AS]=seen[ques[CANDIDATE]]
                    

            hitjson[CANDIDATES] = cand_questions

            outstr = json.dumps(hitjson)
            csvwriter.writerow([idx, outstr.replace("\'", "\\'")])
            print("Wrote line {}, for context {}".format(idx, pmi[0]))
        outwriter.close()



                
            


        



