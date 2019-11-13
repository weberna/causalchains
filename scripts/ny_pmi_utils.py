import sys
import math
import pickle
import numpy as np
import json
from scipy import sparse

def load_vocab(filename):
    #load vocab from json file
    with open(filename, 'rb') as fi:
        voc = pickle.load(fi)
    return voc


def pmi_top_k(cx_counts, cxy_counts, evocab, k=35):
    cx_total = sum([x[1] for x in cx_counts.items()])
    cxy_total = sum([x[1] for x in cxy_counts.items()])
    pmi_dict = {} #pmi dict should be a dict mapping an event to [[first event, logpmi], [first event, logpmi], ...]
    #Pmi maps event to its most likely previous events

    print("C_x total: {}".format(cx_total))
    print("C_xy total: {}".format(cxy_total))

    cxy_counts = dict([(x[0], x[1]) for x in cxy_counts.items() if x[0][0] in evocab.stoi and x[0][1] in evocab.stoi]) #filter counts
    #cx_counts = dict([(x[0], x[1]) if x[1] > 100 else (x[0], 0) for x in cx_counts.items()])

    print("Done Filtering Counts")

    for i, e2 in enumerate(evocab.itos): #get counts only for stuff in our vocab
        if e2 in cx_counts and e2 not in pmi_dict:
            log_pmi = calculate_event_pmi(e2, cx_counts, cxy_counts, cx_total, cxy_total, evocab)
            #log_pmi = calculate_event_sym_pmi(e2, cx_counts, cxy_counts, cx_total, cxy_total, evocab)
            log_pmi = sorted(log_pmi, reverse=True, key=lambda x: x[1])[:k]
            pmi_dict[e2] = log_pmi
            if i % 100 == 0:
                print("Processed {}".format(i))
    return pmi_dict
            
            

def calculate_event_pmi(e2, cx_counts, cxy_counts, cx_total, cxy_total, evocab): #return list of tuples [(first event, logpmi), ...]
    pmis = []
    c_e2 = cx_counts[e2]
    for i, e1 in enumerate(evocab.itos):
        if e1 in cx_counts and e1 != e2:
            c_e1 = cx_counts[e1]
            query = (e1, e2)
     #       if query in cxy_counts and c_e1 > 0 and c_e2 > 0 and cxy_counts[query] > 0:
            if query in cxy_counts and c_e1 > 100 and c_e2 > 0 and cxy_counts[query] > 0:
                #logpmi = math.log(cxy_counts[query]/cxy_total) - math.log(c_e2/cx_total) - math.log(c_e1/cx_total)
                logpmi = math.log(cxy_counts[query]) - math.log(c_e2) - math.log(c_e1)
                pmis.append((e1, logpmi))
    return pmis

def calculate_event_sym_pmi(e2, cx_counts, cxy_counts, cx_total, cxy_total, evocab): #return list of tuples [(first event, logpmi), ...]
    pmis = []
    c_e2 = cx_counts[e2]
    for i, e1 in enumerate(evocab.itos):
        if e1 in cx_counts and e1 != e2:
            c_e1 = cx_counts[e1]
            count = 0
            query = (e1, e2)
            revquery = (e2, e1)

            if query in cxy_counts:
                count += cxy_counts[query]
            if revquery in cxy_counts:
                count += cxy_counts[revquery]

            if c_e1 > 0 and c_e2 > 0 and count > 0:
                #logpmi = math.log(cxy_counts[query]/cxy_total) - math.log(c_e2/cx_total) - math.log(c_e1/cx_total)
                logpmi = math.log(count) - math.log(c_e2) - math.log(c_e1)
                pmis.append((e1, logpmi))
    return pmis


if __name__=="__main__":
    cx_file = sys.argv[1]
    cxy_file = sys.argv[2]
    outfile = sys.argv[3]
    evocab_file = sys.argv[4]

    with open(cx_file, 'rb') as fi:
        upickler = pickle._Unpickler(fi)
        upickler.encoding = 'latin1'
        cx_counts = upickler.load()

    with open(cxy_file, 'rb') as fi:
        upickler = pickle._Unpickler(fi)
        upickler.encoding = 'latin1'
        cxy_counts = upickler.load()

    evocab = load_vocab(evocab_file)

    pmi_dict = pmi_top_k(cx_counts, cxy_counts, evocab)

    with open(outfile, 'w') as outfi:
        json.dump(pmi_dict, outfi)


#pmi should be log(count(x, y)/#of bigrams) - log(c_x/unigramtotal) -log(c_y/unigramtotal)
#use 'some' pickle files (not 'sample')

#Cx loads as a data file dict mapping predicate to count
#Cxy loads dict mapping tuple (e1, e2) to count
