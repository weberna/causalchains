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

def top_e1_choices(e2, scores, evocab):
    """
    return back a list of the top choices according to the causal model
    for a e1 that would explain e2. 
    params:
        (str) e2 : a string representation of the second event
        scores : the output of causalchains.train.testing.normalized_intervention_mat, a tuple
                 with (interven_dists, so_events_itos, so_events_stoi), where intervention dist is a matrix
                 whose jth column is a normalized selection of potential causes of e2
        evocab: The original event vocabulary
    """
    tops = sorted([(scores[1][i], x) for i, x in enumerate(torch.Tensor(scores[0])[:, evocab.stoi[e2]].tolist())], key=lambda l:l[1], reverse=True)
    return [(x[0][1], x[1]) for x in tops]


def top_k_logits(logits, k): #zero out everything except the top k
    vals,_=torch.topk(logits,k)
    mins = vals[:,-1].unsqueeze(dim=1).expand_as(logits)
    return torch.where(logits < mins, torch.ones_like(logits)*-1e10, logits)

def usable(e2, e1):
    """
    Return true is e1 is a usuable output for e1, use this for all models 

    Removes repeats and events without nsubj dobj or iobj
    """

    #Various 'mispelled' words that might be confusing for annotation, just remove them
    nongrammatical = ['thinkin', 'talkin', 'sayin', 'doin', 'lookin', 'gettin', 'tellin', 'fuckin', 'askin', 'walkin', 'makin', 'feelin', 'leavin', 'comin', 'sittin', 'stayin', 'wonderin', 'seein', 'drivin', 'playin']
    modals = ['didnt', 'dont', 'wouldnt', 'couldnt', 'shouldnt', 'didna', 'hadnt', 'doesnt', 'wasnt', 'canna']

    if e2 == e1:
        return False
    if e1.split('->')[0] == e2.split('->')[0]:
        return False
    if e1.split('->')[1] not in ['nsubj', 'dobj', 'iobj']:
        return False
    if len(e1.split('->')) > 2:
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


def complete_chains_lm(model, example, evocab, topk=5, max_len=20):
    """
    Complete generation of an event chain given prefix example
    params:
        causalchains.models.LM (model)
        example:  List[seqlens] - List of the (string form) of events, to be converted to readable inputs here
    """

    model.eval()
    outputs = list(example)

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
    for i in range(max_len):
        already_used = [evocab.stoi[x] for x in outputs] + [evocab.stoi['<unk>']]
        for idx in already_used:
            logits[0, idx] = -1e10

        top_logits = top_k_logits(logits, topk)
        probs = F.softmax(top_logits, dim=1)
        top_inds = torch.multinomial(probs, 1)
        outputs.append(evocab.itos[top_inds.squeeze().item()])

        prev_output=top_inds.squeeze().view(1,-1)
        if evocab.itos[top_inds.squeeze().item()] == EOS_TOK:
            break
        else:
            logits, hidden = model(prev_output, hidden) 
    return outputs

def pairwise_lm_prob(model, example, evocab, topk=300):
    """
    Complete generation of an event chain given prefix example
    params:
        causalchains.models.LM (model)
        example:  (string form) of event
    """

    model.eval()
    outputs = []

    #Process the prefix
    text_inst= torch.LongTensor([evocab.stoi[SOS_TOK]] + [evocab.stoi[example]]) #seqlen tensor]
    
    hidden=None
    logits = None
    for step in range(text_inst.shape[0]):
        print("INPUTTINN {}".format(evocab.itos[text_inst[step]]))
        step_inp = text_inst[step] #get all instances for this step
        step_inp = step_inp.unsqueeze(0).unsqueeze(0) #[1 X 1]

        logit_i, hidden = model(step_inp, hidden)
        logits = logit_i #[1 X vocab]

    #decode
    already_used = [evocab.stoi[example]] + [evocab.stoi['<unk>']]
    for idx in already_used:
        logits[0, idx] = -1e10

    logits = logits.squeeze(dim=0) #[vocab]
    topinds = logits.topk(topk)[1]
    return [evocab.itos[x] for x in topinds]


def pairwise_chains(e2_list, pmi_dict, causal_dict, evocab, evocab_lm=None, lm_dict=None):
    """
    Return back the top candidates for e1 for each item in list e2_list
    params:
        lists of tuples (e2, [list of events in str form])
        
    """
    pmi_res = []
    causal_res = []
    lm_res = []

    for idx, e2 in enumerate(e2_list):
        top_pmi = [x[0] for x in pmi_dict[e2] if usable(e2, x[0]) and x[0] in evocab.itos and x[0] not in evocab.itos[:22]]
        top_causal = [x[0] for x in top_e1_choices(e2, causal_dict, evocab) if usable(e2, x[0]) and x[0] not in evocab.itos[:22]]
        if evocab_lm is not None:
            top_lm = [x[0] for x in top_e1_choices(e2, lm_dict, evocab_lm) if usable(e2, x[0]) and x[0] not in evocab.itos[:22]]
            lm_res.append((e2, top_lm[:2]))

        pmi_res.append((e2, top_pmi[:2]))
        causal_res.append((e2, top_causal[:2]))
        print("Processed event {}: {}".format(idx,e2))

    return (pmi_res, causal_res, lm_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CreateCandidates')
    parser.add_argument('--pmi_dict', type=str, help='pmi json information')
    parser.add_argument('--evocab', type=str, help='the event vocabulary pickle file', default='./data/evocab_freq25')
    parser.add_argument('--causal_dict', type=str, help='Matrix output of causal model, output of causalchains.train.testing.normalized_score_matrix')
    parser.add_argument('--bad_words', type=str, default='data/bad-words.txt')
    parser.add_argument('--candidates', type=str, default='data/e2_candidates_list.txt')
    parser.add_argument('--turk_format', action='store_true')
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--lm_dict', type=str, default=None)
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


    if args.lm_dict is not None:
        evocab_lm = du.convert_to_lm_vocab(copy.deepcopy(evocab))
        with open(args.lm_dict, 'rb') as fi:
            lm_dict = pickle.load(fi)
    else:
        evocab_lm = None
        lm_dict = None

#    e2_cands = [x for x in evocab.itos if len(x.split('->')) == 2 and x.split('->')[1] in ['nsubj', 'dobj']]

    e2_cands = []  
    print("Using Candidate Premises in {}".format(args.candidates))
    with open(args.candidates, 'r') as fi:
        for line in fi:
            event = line.split(',')[0].strip()
            e2_cands.append(event)

    pmi_res, causal_res, lm_res = pairwise_chains(e2_cands, pmi_dict, causal_dict, evocab, evocab_lm, lm_dict)


    if not args.turk_format:
        with open(args.outfile, 'w') as fi:
            if args.lm_dict is not None:
                for idx, (pmi, causal, lm) in enumerate(zip(pmi_res, causal_res, lm_res)):
                    assert pmi[0] == causal[0] == lm[0]
                    fi.write("Event 2: {}\n".format(pmi[0]))
                    fi.write("\tPMI choices for Event 1: {}\n".format(",".join(pmi[1])))
                    fi.write("\tCausal choices for Event 1: {}\n".format(",".join(causal[1])))
                    fi.write("\tCausal choices for Event 1: {}\n".format(",".join(lm[1])))
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
            protag = random.choice(names)
            hitjson = {}
            hitjson[CONTEXT] = convert_to_text(pmi[0], protag, conj_pred=False)
            cand_questions = []

            if len(pmi[1]) != 2 or len(causal[1]) != 2 or len(lm[1]) != 2:
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



                
            


        



