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
import causalchains.utils.data_utils as du
from causalchains.utils.data_utils import PAD_TOK, EOS_TOK, SOS_TOK
from causalchains.models import LM
import time
from torchtext.vocab import GloVe
import pickle
import gc
import glob
import sys
import os
import logging
from causalchains.train.masked_cross_entropy import masked_cross_entropy

def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)



def check_save_model_path(save_model):
    save_model_path = os.path.abspath(save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def validation(args, val_batches, model):
    model.eval()

    valid_loss = 0.0
    with torch.no_grad():
        for iteration, inst in enumerate(val_batches): 
            instance = du.lm_send_instance_to(inst, args.device)

            text_inst, text_lens = inst.text
            target_inst, target_lens = inst.target

            logits = []
            hidden=None
            for step in range(text_inst.size(1)):
                step_inp = text_inst[:, step] #get all instances for this step
                step_inp = step_inp.unsqueeze(1) #[batch X 1]

                logit_i, hidden = model(step_inp, hidden)
                logits += [logit_i]
            logits = torch.stack(logits, dim=1) #[batch, seq_len, vocab]

            loss = masked_cross_entropy(logits, target_inst, target_lens)
            valid_loss+=loss.cpu()
    valid_loss = valid_loss/(iteration+1)

    return valid_loss


def train(args):
    """
    Train the model in the ol' fashioned way, just like grandma used to
    Args
        args (argparse.ArgumentParser)
    """
    #Load the data
    logging.info("Loading Vocab")
    evocab = du.load_vocab(args.evocab)
    logging.info("Event Vocab Loaded, Size {}".format(len(evocab.stoi.keys())))

    evocab.stoi[SOS_TOK] = len(evocab.itos)
    evocab.itos.append(SOS_TOK)

    evocab.stoi[EOS_TOK] = len(evocab.itos)
    evocab.itos.append(EOS_TOK)

    assert evocab.stoi[EOS_TOK] == evocab.itos.index(EOS_TOK) 
    assert evocab.stoi[SOS_TOK] == evocab.itos.index(SOS_TOK) 

    if args.load_model:
        logging.info("Loading the Model")
        model = torch.load(args.load_model, map_location=args.device)
    else:
        logging.info("Creating the Model")
        model = LM.EventLM(args.event_embed_size, args.rnn_hidden_dim, args.rnn_layers, len(evocab.itos), dropout=args.dropout)

    model = model.to(device=args.device)

    #create the optimizer
    if args.load_opt:
        logging.info("Loading the optimizer state")
        optimizer = torch.load(args.load_opt)
    else:
        logging.info("Creating the optimizer anew")
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr)
      #  optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)

    logging.info("Loading Datasets")

    train_dataset = du.LmInstanceDataset(args.train_data, evocab) 
    valid_dataset = du.LmInstanceDataset(args.valid_data, evocab)

    #Remove UNK events from the e1prev_intext attribute so they don't mess up avg encoders
  #  train_dataset.filter_examples(['e1prev_intext'])  #These take really long time! Will have to figure something out...
  #  valid_dataset.filter_examples(['e1prev_intext'])
    logging.info("Finished Loading Training Dataset {} examples".format(len(train_dataset)))
    logging.info("Finished Loading Valid Dataset {} examples".format(len(valid_dataset)))

    train_batches = BatchIter(train_dataset, args.batch_size, sort_key=lambda x:len(x.text), train=True, repeat=False, shuffle=True, sort_within_batch=True, device=None)
    valid_batches = BatchIter(valid_dataset, args.batch_size, sort_key=lambda x:len(x.text), train=False, repeat=False, shuffle=False, sort_within_batch=True, device=None)
    train_data_len = len(train_dataset)
    valid_data_len = len(valid_dataset)


    start_time = time.time() #start of epoch 1
    best_valid_loss= float('inf')
    best_epoch = args.epochs 

    #MAIN TRAINING LOOP
    for curr_epoch in range(args.epochs):
        prev_losses = []
        for iteration, inst in enumerate(train_batches): 
            instance = du.lm_send_instance_to(inst, args.device)

            text_inst, text_lens = inst.text
            target_inst, target_lens = inst.target

            model.train()
            model.zero_grad()

            logits = []
            hidden=None
            for step in range(text_inst.size(1)):
                step_inp = text_inst[:, step] #get all instances for this step
                step_inp = step_inp.unsqueeze(1) #[batch X 1]

                logit_i, hidden = model(step_inp, hidden)
                logits += [logit_i]
            logits = torch.stack(logits, dim=1) #[batch, seq_len, vocab]

            loss = masked_cross_entropy(logits, target_inst, target_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step() 

            prev_losses.append(loss.cpu().data)
            prev_losses = prev_losses[-50:]

            if (iteration % args.log_every == 0) and iteration != 0:
                past_50_avg = sum(prev_losses) / len(prev_losses)
                logging.info("Epoch/iteration {}/{}, Past 50 Average Loss {}, Best Val {} at Epoch {}".format(curr_epoch, iteration, past_50_avg, 'NA' if best_valid_loss == float('inf') else best_valid_loss, 'NA' if best_epoch == args.epochs else best_epoch))

            if (iteration % args.validate_after == 0) and iteration != 0:
                logging.info("Running Validation at Epoch/iteration {}/{}".format(curr_epoch, iteration))
                new_valid_loss = validation(args, valid_batches, model)
                logging.info("Validation loss at Epoch/iteration {}/{}: {:.3f} - Best Validation Loss: {:.3f}".format(curr_epoch, iteration, new_valid_loss, best_valid_loss))
                if new_valid_loss < best_valid_loss:
                    logging.info("New Validation Best...Saving Model Checkpoint")  
                    best_valid_loss = new_valid_loss
                    best_epoch = curr_epoch
                    #torch.save(model, "{}.epoch_{}.loss_{:.2f}.pt".format(args.save_model, curr_epoch, best_valid_loss))
                    #torch.save(optimizer, "{}.{}.epoch_{}.loss_{:.2f}.pt".format(args.save_model, "optimizer", curr_epoch, best_valid_loss))
                    torch.save(model, "{}".format(args.save_model))
                    torch.save(optimizer, "{}_optimizer".format(args.save_model))

        #END OF EPOCH
        logging.info("End of Epoch {}, Running Validation".format(curr_epoch))
        new_valid_loss = validation(args, valid_batches, model)
        logging.info("Validation loss at end of Epoch {}: {:.3f} - Best Validation Loss: {:.3f}".format(curr_epoch, new_valid_loss, best_valid_loss))
        if new_valid_loss < best_valid_loss:
            logging.info("New Validation Best...Saving Model Checkpoint")  
            best_valid_loss = new_valid_loss
            best_epoch = curr_epoch
            #torch.save(model, "{}.epoch_{}.loss_{:.2f}.pt".format(args.save_model, curr_epoch, best_valid_loss))
            #torch.save(optimizer, "{}.{}.epoch_{}.loss_{:.2f}.pt".format(args.save_model, "optimizer", curr_epoch, best_valid_loss))
            torch.save(model, "{}".format(args.save_model))
            torch.save(optimizer, "{}_optimizer".format(args.save_model))

        if curr_epoch - best_epoch >= args.stop_after:
            logging.info("No improvement in {} epochs, terminating at epoch {}...".format(args.stop_after, curr_epoch))
            logging.info("Best Validation Loss: {:.2f} at Epoch {}".format(best_valid_loss, best_epoch))
            break

             


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training for Nuisance Conditionals')
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--evocab', type=str, help='the event vocabulary pickle file', default='./data/evocab_freq25')
    parser.add_argument('--event_embed_size', type=int, default=300, help='size of event embeddings')
    parser.add_argument('--rnn_hidden_dim', type=int, default=512, help='size of mlp hidden layer for component models')
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--log_every', type=int, default=500)
    parser.add_argument('--validate_after', type=int, default=5000)
    parser.add_argument('--optimizer', type=str, default='adam', help='adam, adagrad, sgd')
    parser.add_argument('--clip', type=float, default=10.0, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
    parser.add_argument('--stop_after', type=int, default=3, help='Stop after this many epochs have passed without decrease in validation loss')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=11, help='random seed') 
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('-save_model', default='model_checkpoint.pt', help="""Model filename""")
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--load_opt', type=str)


    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    args.device=None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open('{}_args.pkl'.format(args.save_model), 'wb') as fi:
        pickle.dump(args, fi)

    if torch.cuda.is_available():
        if not args.cuda:
            logging.warning("WARNING: You have a CUDA device, so you should probably run with --cuda")
            args.device = torch.device('cpu')
        else:
            torch.cuda.manual_seed(args.seed)
            args.device = torch.device('cuda')

            logging.info("Using GPU {}".format(torch.cuda.get_device_name(args.device)))

    else:
        args.device = torch.device('cpu')
    

    train(args)



