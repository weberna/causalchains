import sys
import pickle
import numpy as np
import json
from scipy import sparse

def pmi_top_k(pmi_matrix, itos, stoi, k=35):
    pmi_dict = {}
    numwords = pmi_matrix.shape[0]
    assert numwords == len(itos) == len(stoi.keys())
    for i in range(numwords):
        row = pmi_matrix[i, :].toarray()[0] #numwords len numpy array
        e1 = itos[i]
        verb = e1.split('->')[0]
        cooccurs = [(itos[idx], pmi) for idx, pmi in enumerate(row)]
        cooccurs = [x for x in cooccurs if x[1] < 0.0]
        cooccurs = [x for x in cooccurs if x[0].split('->')[0] != verb] #dont take duplicate predicates
        cooccurs = sorted(cooccurs, reverse=True, key=lambda x: x[1])[:k] #take top k
        assert e1 not in pmi_dict
        pmi_dict[e1] = cooccurs
        if i % 100 == 0:
            print("Processed {}".format(i))
    return pmi_dict
        


if __name__=="__main__":
    pmi_dump = sys.argv[1]
    outfile = sys.argv[2]

    with open(pmi_dump, 'rb') as fi:
        pmi_tup = pickle.load(fi)

    pmi_matrix, _, _, voc = pmi_tup

    stoi = dict([(x[1], x[0]) for x in enumerate(voc)])
    itos = voc

    pmi_dict = pmi_top_k(pmi_matrix, itos, stoi)

    with open(outfile, 'w') as outfi:
        json.dump(pmi_dict, outfi)

