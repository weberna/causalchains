import torch 
import torch.nn as nn
from torchtext.data import Iterator as BatchIter
import argparse
import numpy as np
import random
import math
import torch.nn.functional as F
import causalchains.utils.data_utils as du
from causalchains.utils.data_utils import PAD_TOK
import causalchains.models.estimator_model as estimators
import time
from torchtext.vocab import GloVe
import pickle
import gc
import glob
import sys
import os
import logging

from causalchains.models.estimator_model import EXP_OUTCOME_COMPONENT, PROPENSITY_COMPONENT

def prediction_topk(model_outputs, evocab, k=5):
    output = model_outputs[EXP_OUTCOME_COMPONENT]
    val, idx = output.topk(k, dim=1)
    idx = idx[0]
    predicted = [evocab.itos[x] for x in idx]
    return predicted


def intervention_dist(dset, model, e1, evocab, device=None, batch_size=1024):
    accum = torch.zeros(len(evocab.itos))
    batches = [dset.examples[x:x+batch_size] for x in range(0, len(dset.examples), batch_size)]
    with torch.no_grad():
        for batch in batches:
            for example in batch:
                example.e1 = e1 #e1 is a string
            output=model(dset.example_to_batch(batch, device=device, multiple=True))
            output = output[EXP_OUTCOME_COMPONENT] #logits, batch X dim
            sm_output = F.softmax(output, dim=1)
            accum += torch.sum(sm_output, dim=0).cpu()

    final = accum / len(dset.examples)
    return final #tensor[vocab_len]



def normalized_scores_matrix(args, model):
    evocab = du.load_vocab(args.evocab)
    tvocab = du.load_vocab(args.tvocab)
    outfile = args.outfile
    min_size = model.text_encoder.largest_ngram_size #Add extra pads if text size smaller than largest CNN kernel size
    dset= du.InstanceDataset(args.data, evocab, tvocab, min_size=min_size) 

    events = evocab.stoi.keys()
    #so_events = [x for x in events if len(x.split('->'))==2 and x.split('->')[1] in ['nsubj', 'dobj', 'iobj']]
    so_events = events

    so_events_itos = list(enumerate(so_events))
    so_events_stoi = dict([(x[1], x[0]) for x in so_events_itos])

    interven_dists = []

    for i, e1 in so_events_itos:
        logging.info("Computing Intervention for Event {}, {}".format(i, e1))
        interven_e1 = intervention_dist(dset, model, e1, evocab, device=args.device)
        interven_dists.append(interven_e1)

    interven_dists = torch.stack(interven_dists, dim=0)
    normalizer = torch.sum(interven_dists, dim=0).unsqueeze(dim=0)
    interven_dists = interven_dists / normalizer
    interven_dists = interven_dists.tolist() #so_events_itos len X evocab len
    output = (interven_dists, so_events_itos, so_events_stoi)
    with open(outfile, 'wb') as fi:
        pickle.dump(output, fi)


def ate_matrix(args, model, baseline='say->nsubj'):
    evocab = du.load_vocab(args.evocab)
    tvocab = du.load_vocab(args.tvocab)
    outfile = args.outfile
    min_size = model.text_encoder.largest_ngram_size #Add extra pads if text size smaller than largest CNN kernel size
    dset= du.InstanceDataset(args.data, evocab, tvocab, min_size=min_size) 

    events = evocab.stoi.keys()
    so_events = [x for x in events if len(x.split('->'))==2 and x.split('->')[1] in ['nsubj', 'dobj', 'iobj']]
    #so_events = events

    so_events_itos = list(enumerate(so_events))
    so_events_stoi = dict([(x[1], x[0]) for x in so_events_itos])

    interven_dists = []

    baseline_dist = intervention_dist(dset, model, baseline, evocab, device=args.device)

    for i, e1 in so_events_itos:
        logging.info("Computing Intervention for Event {}, {}".format(i, e1))
        interven_e1 = intervention_dist(dset, model, e1, evocab, device=args.device)
        interven_dists.append(interven_e1 - baseline_dist) #ith element contains effect of do(e1) on ith vocab element

    interven_dists = torch.stack(interven_dists, dim=0)
    interven_dists = interven_dists.tolist() #so_events_itos len X evocab len
    output = (interven_dists, so_events_itos, so_events_stoi)
    with open(outfile, 'wb') as fi:
        pickle.dump(output, fi)


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--data', type=str)
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--evocab', type=str, help='the event vocabulary pickle file', default='./data/evocab_freq25')
    parser.add_argument('--tvocab', type=str, help='the text vocabulary pickle file', default='./data/tvocab_freq100')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')

    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    args.device = None


    if torch.cuda.is_available():
        if not args.cuda:
            logging.warning("WARNING: You have a CUDA device, so you should probably run with --cuda")
            args.device = torch.device('cpu')
        else:
            #torch.cuda.manual_seed(args.seed)
            args.device = torch.device('cuda')

            logging.info("Using GPU {}".format(torch.cuda.get_device_name(args.device)))

    else:
        args.device = torch.device('cpu')


    model = torch.load(args.model, map_location=args.device)
    model.eval()
    ate_matrix(args, model)

    



            

    
