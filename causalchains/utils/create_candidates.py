import sys 
import os
import json
import argparse
import causalchains.utils.data_utils as du
from collections import namedtuple

STOP_EVENTS = ['go', 'do', 'have']
Line = namedtuple('Line', ['e1', 'e1_pred', 'e1_rel', 'other_events', 'other_events_pred', 'other_events_rel'])

#All dicts map event -> list of tuples (possible prev event, score)
def load_dicts(args):
    with open(args.conceptnet_json, 'r') as fi:
        cn_dict = json.load(fi)

    with open(args.verbocean_json, 'r') as fi:
        vo_dict = json.load(fi)

    with open(args.toronto_pmi_json, 'r') as fi:
        tor_pmi_dict = json.load(fi)

    return cn_dict, vo_dict, tor_pmi_dict

def process_line(line):
    e1 = line['e1']
    e1_pred = e1.split('->')[0]
    e1_rel = "->".join(e1.split('->')[1:])
    other_events = line['e1prev_intext'] + [line['e2']]
    other_events_pred = [x.split('->')[0] for x in other_events]
    other_events_rel = [x.split('->')[1:] for x in other_events]

    return Line(e1, e1_pred, e1_rel, other_events, other_events_pred, other_events_rel)

def in_vocab(p, evocab, rel):
    if p+'->nsubj' in evocab.stoi or p+'->dobj' in evocab.stoi or p+'->arg' in evocab.stoi or p+'->iobj' in evocab.stoi or p+rel in evocab.stoi:
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
            for p in preds:
                single_preds.append((p, edge[1]))
    edges = [x for x in edges if len(x[0].split(" ")) == 1] #remove multiword predicates
    edges.extend(single_preds) #add back in the valid singular versions
    edges = [i for n, i in enumerate(edges) if i not in edges[:n]] #remove duplicates
    return edges
                
def get_conceptnet_cands(line: Line, cn_dict, evocab, num_cands=2):
    if line.e1_pred not in cn_dict:
        return []

    edges = [x for x in cn_dict[line.e1_pred] if x[0] != line.e1_pred] #Get edges (non duplicates)
    edges = handle_multi_word(edges, evocab)
    edges = [x for x in edges if x[0] not in STOP_EVENTS]
    edges = sorted(edges, reverse=True, key=lambda x: x[1])
    cands = [x[0] for x in edges[:num_cands]]
    cands = [x for x in cands if in_vocab(x, evocab, line.e1_rel)]
    return cands

def get_verbocean_cands(line: Line, vo_dict, evocab, num_cands=2):
    if line.e1_pred not in vo_dict:
        return []

    edges = [x for x in vo_dict[line.e1_pred] if x[0] != line.e1_pred] #Get edges (non duplicates)
    edges = [x for x in edges if x[0] not in STOP_EVENTS]
    edges = sorted(edges, reverse=True, key=lambda x: x[1])
    cands = [x[0] for x in edges[:num_cands]]
    cands = [x for x in cands if in_vocab(x, evocab, line.e1_rel)]
    return cands

def get_pmi_cands(line: Line, pmi_dict, evocab):
    if line.e1 not in pmi_dict:
        return []

    edges = [x for x in pmi_dict[line.e1] if x[0] != line.e1] #Get edges (non duplicates)
    edges = [x for x in edges if x[0].split('->')[0] not in STOP_EVENTS]
    edges = sorted(edges, reverse=True, key=lambda x: x[1])
    cands = [x[0] for x in edges]
    return cands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CreateCandidates')
    parser.add_argument('--origdata', type=str, help='the original dataset without candidates')
    parser.add_argument('--evocab', type=str, help='event vocab file')
    parser.add_argument('--newdata', type=str, help='Where do write the new dataset with candidates')
    parser.add_argument('--conceptnet_json', type=str, help='Concept net queries output' )
    parser.add_argument('--verbocean_json', type=str, help='Concept net queries output' )
    parser.add_argument('--toronto_pmi_json', type=str, help='Concept net queries output' )
    parser.add_argument('--total_cands', type=int, default=6)
    args = parser.parse_args()

    cn_dict, vo_dict, tor_pmi_dict = load_dicts(args)
        
    evocab = du.load_vocab(args.evocab)
    origfi = open(args.origdata, 'r')
    outwriter = open(args.newdata, 'w')

    for i, line in enumerate(origfi):
        js_line = json.loads(line)
        instance = process_line(js_line)

        cn_cands = get_conceptnet_cands(instance, cn_dict, evocab)
        cn_cands = [(x, 'ConceptNet') for x in cn_cands]
        
        vo_cands = get_verbocean_cands(instance, vo_dict, evocab)
        vo_cands = [(x, 'VerbOcean') for x in vo_cands]

        pmi_cands = get_pmi_cands(instance, tor_pmi_dict, evocab)
        pmi_cands = [(x, 'PMI') for x in pmi_cands]

        candidates = cn_cands + vo_cands + pmi_cands
        candidates = candidates[:args.total_cands]

        js_line['cands'] = candidates

        json.dump(js_line, outwriter)
        outwriter.write("\n")
        print("Processed instance {}, Total Candidates: {}, CN: {}, VO: {}, PMI: {}".format(i, 
                                                                                            len(candidates),
                                                                                            len(cn_cands),
                                                                                            len(vo_cands),
                                                                                            len(pmi_cands)))

    origfi.close()
    outwriter.close()
       


            


    


            
            
