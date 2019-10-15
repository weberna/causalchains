########################################
#   module for training necessary components
#   of the model
########################################
import torch 
import torch.nn as nn
from torchtext.data import Iterator as BatchIter
import argparse
import numpy as np
import random
import math
import torch.nn.functional as F
import causalchains.data_utils as du
from causalchains.data_utils import PAD_TOK
import causalchains.models.estimator_model as estimators
import time
from torchtext.vocab import GloVe
import pickle
import gc
import glob
import sys
import os

from estimators import EXP_OUTCOME_COMPONENT, PROPENSITY_COMPONENT


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)



def check_save_model_path(save_model):
    save_model_path = os.path.abspath(save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


        
def train(args):
    """
    Train the model in the ol' fashioned way, just like grandma used to
    Args
        args (argparse.ArgumentParser)
    """
    #Load the data
    print("\nLoading Vocab")
    evocab = du.load_vocab(args.evocab)
    tvocab = du.load_vocab(args.tvocab)
    print("Event Vocab Loaded, Size {}".format(len(evocab.stoi.keys())))
    print("Text Vocab Loaded, Size {}".format(len(tvocab.stoi.keys())))


    print("Loading Dataset")
    train_dataset = du.InstanceDataset(args.train_data, evocab, tvocab) 
    print("Finished Loading Dataset {} examples".format(len(train_dataset)))
    train_batches = BatchIter(train_dataset, args.batch_size, sort_key=lambda x:len(x.e1_text), train=True, sort_within_batch=True, device=-1)
    train_data_len = len(train_dataset)

    if args.load_model:
        print("Loading the Model")
        model = torch.load(args.load_model)
    else:
        print("Creating the Model")
        model = DAVAE(args.emb_size, hidsize, vocab, latents, layers=args.nlayers, use_cuda=use_cuda, pretrained=args.use_pretrained, dropout=args.dropout)
        model = estimators.NaiveAdjustmentEstimator(args, evocab, tvocab)

    #create the optimizer
    if args.load_opt:
        print("Loading the optimizer state")
        optimizer = torch.load(args.load_opt)
    else:
        print("Creating the optimizer anew")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_func = nn.CrossEntropyLoss()

    start_time = time.time() #start of epoch 1
    curr_epoch = 1
    valid_loss = [0.0]
    for iteration, instance in enumerate(train_batches): #this will continue on forever (shuffling every epoch) till epochs finished
        model.zero_grad()

        model_outputs = model(instance) 

        exp_outcome_out = model_outputs[EXP_OUTCOME_COMPONENT]  #[batch X num events], output predication for e2
        exp_outcome_loss = loss_func(exp_outcome_out, instance.e2)

        loss = exp_outcome_loss
        # backward propagation
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        # Optimize
        optimizer.step() 

        # End of an epoch - run validation

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DAVAE')
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--evocab', type=str, help='the event vocabulary pickle file')
    parser.add_argument('--tvocab', type=str, help='the text vocabulary pickle file')
    parser.add_argument('--event_embed_size', type=int, default=32, help='size of event embeddings')
    parser.add_argument('--text_embed_size', type=int, default=32, help='size of text embeddings')
    parser.add_argument('--text_enc_output', type=int, default=32, help='size of output of text encoder')
    parser.add_argument('--mlp_hidden_dim', type=int, default=32, help='size of mlp hidden layer for component models')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--log_every', type=int, default=200)
    parser.add_argument('--save_after', type=int, default=500)
    parser.add_argument('--validate_after', type=int, default=2500)
    parser.add_argument('--optimizer', type=str, default='adam', help='adam, adagrad, sgd')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=11, help='random seed') 
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('-save_model', default='model_checkpoint.pt', help="""Model filename""")
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--load_opt', type=str)
    parser.add_argument

    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open('{}_args.pkl'.format(args.save_model), 'wb') as fi:
        pickle.dump(args, fi)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    train(args)



