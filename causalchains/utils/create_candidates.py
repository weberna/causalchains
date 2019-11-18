import sys 
import os
import csv
import json
import argparse
import causalchains.utils.data_utils as du
from pattern.en import conjugate
from collections import namedtuple

CANDIDATE_STOP_EVENTS = ['be', 'go', 'do', 'have'] #Stop events for candidates
STOP_EVENTS = ['be', 'go', 'do', 'say', 'know', 'feel', 'want', 'tell', 'wish', 'think', 'start', 'answer', 'whisper'] #Stop events for e1
Line = namedtuple('Line', ['e1', 'e1_pred', 'e1_rel', 'other_events', 'other_events_pred', 'other_events_rel', 'e1arg'])

BAD_WORDS = []

#All dicts map event -> list of tuples (possible prev event, score)
def load_dicts(args):
    with open(args.conceptnet_json, 'r') as fi:
        cn_dict = json.load(fi)

    with open(args.verbocean_json, 'r') as fi:
        vo_dict = json.load(fi)

    with open(args.toronto_pmi_json, 'r') as fi:
        tor_pmi_dict = json.load(fi)

    with open(args.nyt_pmi_json, 'r') as fi:
        nyt_pmi_dict = json.load(fi)


    return cn_dict, vo_dict, tor_pmi_dict, nyt_pmi_dict


def skip_line(js_line): #Whether or not to skip this instance, for various reasons
    global BAD_WORDS
    MINLEN=3
    MAXLEN = 25
    instance = process_line(js_line)
    e1_arg = instance.e1arg
    e1_rel = instance.e1_rel
    e1_pred = instance.e1_pred
    e1_text = js_line['e1_text']
    e1_text_len = len([x for x in e1_text.split(" ") if "'" not in x and "," not in x])

    e1_text_tok = [x.lower() for x in e1_text.split(" ")]

    if any([x in e1_text_tok for x in BAD_WORDS]):
        return True

    if e1_rel not in ['nsubj', 'dobj', 'iobj']:
        return True

    if e1_text_len < MINLEN or e1_text_len > 25:
        return True

    if e1_pred in STOP_EVENTS:
        return True

    if '`' in e1_text or '(' in e1_text or ')' in e1_text or '*' in e1_text or "''" in e1_text:
        return True
        
    if '`' in e1_pred or '(' in e1_pred or ')' in e1_pred or '*' in e1_pred or "'" in e1_pred:
        return True

    return False

def skip_candidate(cand): #Pass in predicate only
    global BAD_WORDS
    if len(cand) < 3: 
        return True
    if '*' in cand or "'" in cand or "`" in cand or '"' in cand:
        return True
    if any([x in cand.lower() for x in BAD_WORDS]):
        return True

    return False
            

def process_line(line):
    e1 = line['e1']
    e1_pred = e1.split('->')[0]
    e1_rel = "->".join(e1.split('->')[1:])
    other_events = line['e1prev_intext'] + [line['e1']] + [line['e2']]
    other_events_pred = [x.split('->')[0] for x in other_events]
    other_events_rel = [x.split('->')[1:] for x in other_events]
    e1arg = line['e1arg']

    return Line(e1, e1_pred, e1_rel, other_events, other_events_pred, other_events_rel, e1arg)

def in_vocab(p, evocab, rel, only_so=True):
    if p+'->nsubj' in evocab.stoi or p+'->dobj' in evocab.stoi or p+'->iobj' in evocab.stoi or (not only_so and p+'->arg' in evocab.stoi) or (not only_so and p+rel in evocab.stoi):
        return True
    else:
        return False



def handle_multi_word(edges, evocab): #split multiword predicates into valid single word predicates
    if not edges:
        return edges

    single_preds = []
    for idx, edge in enumerate(edges):
        preds = edge[0].split(" ")
        if len(preds) > 1:
            single_preds.append((preds[0], edge[1])) #only take first word of multi word predicate (this is usually the verb head)
    edges = [x for x in edges if len(x[0].split(" ")) == 1] #remove multiword predicates
    edges.extend(single_preds) #add back in the valid singular versions

    return edges
                
def get_conceptnet_cands(line: Line, cn_dict, evocab, num_cands=2):
    if line.e1_pred not in cn_dict:
        return []

    edges = [x for x in cn_dict[line.e1_pred] if x[0] != line.e1_pred] #Get edges (non duplicates)
    edges = handle_multi_word(edges, evocab)
    edges = [x for x in edges if x[0] not in CANDIDATE_STOP_EVENTS]
    edges = sorted(edges, reverse=True, key=lambda x: x[1])
    cands = [x[0] for x in edges]

    cands = [i for n, i in enumerate(cands) if i not in cands[:n]] #remove duplicates
    cands = [x for x in cands if in_vocab(x, evocab, line.e1_rel)]
    cands = [x for x in cands if x not in line.other_events_pred]
    cands = [x for x in cands if not skip_candidate(x)]
    return cands[:num_cands]

def get_verbocean_cands(line: Line, vo_dict, evocab, num_cands=3):
    if line.e1_pred not in vo_dict:
        return []

    edges = [x for x in vo_dict[line.e1_pred] if x[0] != line.e1_pred] #Get edges (non duplicates)
    edges = [x for x in edges if x[0] not in CANDIDATE_STOP_EVENTS]
    edges = sorted(edges, reverse=True, key=lambda x: x[1])
    cands = [x[0] for x in edges]
    cands = [x for x in cands if in_vocab(x, evocab, line.e1_rel)]
    cands = [x for x in cands if x not in line.other_events_pred]
    cands = [x for x in cands if not skip_candidate(x)]
    return cands[:num_cands]

def get_pmi_cands(line: Line, pmi_dict, evocab, num_cands=None):
    if line.e1 not in pmi_dict:
        return []

    edges = [x for x in pmi_dict[line.e1] if x[0] != line.e1] #Get edges (non duplicates)
    edges = [x for x in edges if x[0].split('->')[0] not in CANDIDATE_STOP_EVENTS and len(x[0].split('->')[0].split(':')) == 1]
    edges = sorted(edges, reverse=True, key=lambda x: x[1])
    cands = [x[0] for x in edges]
    cands = [x for x in cands if x.split('->')[0] not in line.other_events_pred and x.split('->')[1] in ['nsubj', 'dobj', 'iobj']]
    cands = [x for x in cands if not skip_candidate(x.split('->')[0])]

    if num_cands is not None:
        cands = cands[:num_cands]

    return cands

def convert_pronoun(e1_arg, rel):
    new_e1_arg = e1_arg
    if rel == 'nsubj':
        if e1_arg.lower() == "me":
            new_e1_arg = "I"
        elif e1_arg.lower() == "her":
            new_e1_arg = "She"
        elif e1_arg.lower() == "him":
            new_e1_arg = "He"
        elif e1_arg.lower() == "them":
            new_e1_arg = "They"
    elif rel == 'dobj' or rel == 'iobj':
        if e1_arg.lower() == "i":
            new_e1_arg = "me"
        elif e1_arg.lower() == "she":
            new_e1_arg = "her"
        elif e1_arg.lower() == "he":
            new_e1_arg = "him"
        elif e1_arg.lower() == "they":
            new_e1_arg = "them"
    return new_e1_arg



def convert_to_text_old(cand_pred, e1_arg, e1_rel, e1_pred):
    if len(cand_pred.split('->')) == 1: #from concept net or VO
        event = cand_pred + "->" + e1_rel
    else:
        event = cand_pred
        e1_rel = cand_pred.split('->')[1]
        cand_pred = cand_pred.split('->')[0]

    e1_arg = convert_pronoun(e1_arg, e1_rel)
    if e1_rel == 'nsubj':
        #These two exceptions common enough to handle outright
        if (e1_pred == 'see' and cand_pred == 'show') or (e1_pred == 'have' and cand_pred == 'give'): 
            event_text = "Someone" + " " + cand_pred + " something to " + e1_arg
            event = cand_pred + "->iobj"
        else:
            event_text = e1_arg + " " + cand_pred + " " + "(something)"
    elif e1_rel == 'iobj':
        event_text = "Someone" + " " + cand_pred + " " + "something to " + e1_arg
    else:
        event_text = "Someone" + " " + cand_pred + " " + e1_arg

    return event_text.capitalize(), event


def convert_to_text(cand_pred, e1_arg, e1_rel, e1_pred):
    if len(cand_pred.split('->')) == 1: #from concept net or VO
        event = cand_pred + "->" + e1_rel
    else:
        event = cand_pred
        e1_rel = cand_pred.split('->')[1]
        cand_pred = cand_pred.split('->')[0]

    e1_arg = convert_pronoun(e1_arg, e1_rel)

    if e1_arg.lower() == 'you':
        conj = conjugate(cand_pred, '2sgp')
    else:
        conj = conjugate(cand_pred, '1sgp')
    text_cand_pred = conj if conj else cand_pred

    if e1_rel == 'nsubj':
        #These two exceptions common enough to handle outright
        if (e1_pred == 'see' and cand_pred == 'show') or (e1_pred == 'have' and cand_pred == 'give'): 
            event_text = "(Someone or something)" + " " + text_cand_pred + " something to " + e1_arg
            event = cand_pred + "->iobj"
        else:
            event_text = e1_arg + " " + text_cand_pred + " " + "(something or someone)"
    elif e1_rel == 'iobj':
        event_text = "(Someone or something)" + " " + text_cand_pred + " " + "something to " + e1_arg
    else:
        event_text = "(Someone or something)" + " " + text_cand_pred + " " + e1_arg

    return event_text.capitalize(), event


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CreateCandidates')
    parser.add_argument('--origdata', type=str, help='the original dataset without candidates')
    parser.add_argument('--evocab', type=str, help='the event vocabulary pickle file', default='./data/evocab_freq25')
    parser.add_argument('--newdata', type=str, help='Where do write the new dataset with candidates')
    parser.add_argument('--conceptnet_json', type=str, help='Concept net queries output' )
    parser.add_argument('--verbocean_json', type=str, help='Concept net queries output' )
    parser.add_argument('--toronto_pmi_json', type=str, help='Concept net queries output' )
    parser.add_argument('--nyt_pmi_json', type=str, help='Concept net queries output' )
    parser.add_argument('--total_cands', type=int, default=6)
    parser.add_argument('--turk_format', action='store_true')
    parser.add_argument('--bad_words', type=str, default='data/bad-words.txt')
    args = parser.parse_args()

    cn_dict, vo_dict, tor_pmi_dict, nyt_pmi_dict = load_dicts(args)
        
    evocab = du.load_vocab(args.evocab)
    origfi = open(args.origdata, 'r')
    outwriter = open(args.newdata, 'w')

    global BAD_WORDS

    with open(args.bad_words, 'r') as fi:
        bword_lines = fi.readlines()
        BAD_WORDS = [x.rstrip() for x in bword_lines]

    if args.turk_format:
        csvwriter= csv.writer(outwriter)
        csvwriter.writerow(['id', 'json_variables'])

    writtenlines=0

    for i, line in enumerate(origfi):
        js_line = json.loads(line)

        if skip_line(js_line):
            continue

        instance = process_line(js_line)
        e1_arg = instance.e1arg
        e1_rel = instance.e1_rel
        e1_pred = instance.e1_pred

        cn_cands = get_conceptnet_cands(instance, cn_dict, evocab, num_cands=2)
        cn_cands = [(x, convert_to_text(x, e1_arg, e1_rel, e1_pred)[0], convert_to_text(x, e1_arg, e1_rel, e1_pred)[1], "CNET") for x in cn_cands]
        
        vo_cands = get_verbocean_cands(instance, vo_dict, evocab, num_cands=3)
        vo_cands = [(x, convert_to_text(x, e1_arg, e1_rel, e1_pred)[0], convert_to_text(x, e1_arg, e1_rel, e1_pred)[1], "VO") for x in vo_cands]

        nyt_pmi_cands = get_pmi_cands(instance, nyt_pmi_dict, evocab, num_cands=3)
        nyt_pmi_cands = [(x,convert_to_text(x, e1_arg, e1_rel, e1_pred)[0], convert_to_text(x, e1_arg, e1_rel, e1_pred)[1], "NYTPMI") for x in nyt_pmi_cands]

        pmi_cands = get_pmi_cands(instance, tor_pmi_dict, evocab)
        pmi_cands = [(x,convert_to_text(x, e1_arg, e1_rel, e1_pred)[0], convert_to_text(x, e1_arg, e1_rel, e1_pred)[1], "TORPMI") for x in pmi_cands]

        candidates = cn_cands + vo_cands + nyt_pmi_cands + pmi_cands
        candidates = candidates[:args.total_cands]

        #candidates contain tuples (orig candidate event, text, candidate event, source)

        if len(candidates) < args.total_cands:
            continue

        new_js_line = {'e1': js_line['e1'], 'e1_text':js_line['e1_text'], 'cands': candidates, 'id':js_line['id']}

        if not args.turk_format:
            json.dump(new_js_line, outwriter)
            outwriter.write("\n\n")
        else:
            outp = [] 
            for idx, cand in enumerate(candidates):
                turkline = dict(js_line)
                turkline['CANDIDATE'] = cand[1]
                turkline['CANDIDATE_ORIGINAL'] = cand[0]
                turkline['CANDIDATE_EVENT_FORM'] = cand[2]
                turkline['CANDIDATE_SOURCE'] = cand[3]
                turkline['CONTEXT_EVENT'] = convert_to_text(turkline['e1'], e1_arg, e1_rel, '')[0]
                turkline['CONTEXT'] = turkline['e1_text'].capitalize()
                outp.append(turkline)
            outstr = json.dumps(outp)
            csvwriter.writerow([writtenlines, outstr.replace("\'", "\\'")])
            writtenlines += 1


        print("Processed instance {}, Total Candidates: {}, CN: {}, VO: {}, NYT_PMI: {}, PMI: {}".format(i, 
                                                                                            len(candidates),
                                                                                            len(cn_cands),
                                                                                            len(vo_cands),
                                                                                            len(nyt_pmi_cands),
                                                                                            len(pmi_cands)))

    origfi.close()
    outwriter.close()
       


            


    


            
            
