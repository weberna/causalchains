################################################
#   Models for various conditional expectations
################################################
import torch
import torch.nn as nn
import numpy as np
import math
import logging
import causalchains.utils.data_utils as du
from causalchains.utils.data_utils import PAD_TOK
from collections import namedtuple

EmptyEncoder = namedtuple('EmptyEncoder', ['output_dim'])
dummy = EmptyEncoder(0)


class ExpectedOutcome(nn.Module):
    'Models E[e2 | e1, e1_text], Use Embeddings for event representation'

    def __init__(self, event_embeddings, text_embeddings, event_encoder, text_encoder, evocab, tvocab, config):
        """
        Params:
            (Torch.nn.Embeddings) event_embeddings : Pytorch nn.Embeddings Module for events (Pass in None for no event embeddings)
            (Torch.nn.Embeddings) text_embeddings : Pytorch nn.Embeddings Module for text
            (Torch.nn.Module) event_encoder : A module for encodeing events (pass None if not using prev events)
            (Torch.nn.Module) text_encoder : A module for encodeing text 
                    should take in [batch X num tokens X embd dim] Tensor and output [batch X output dim vector]
        """
        super(ExpectedOutcome, self).__init__()

        self.event_embeddings = event_embeddings
        self.text_embeddings = text_embeddings
        self.text_encoder = text_encoder
        self.event_encoder = event_encoder if event_encoder is not None else dummy
        if event_encoder is None:
            logging.info("Not using Event Encoder for Previous Events, Thus, not using Previous Events as input")

        self.event_embed_dim = self.event_embeddings.weight.shape[1] if self.event_embeddings is not None else 0
        self.text_embed_dim = self.text_embeddings.weight.shape[1]
        self.num_events = len(evocab.itos)
        self.hidden_dim = self.event_embed_dim if config.mlp_hidden_dim == 0 else config.mlp_hidden_dim

        self.e_pad = evocab.stoi[PAD_TOK]
        self.t_pad = tvocab.stoi[PAD_TOK]
        self.combine_events = config.combine_events

        if self.combine_events: #Dont give seperate position to e1, treat it like previous context
            logging.info("ExpectedOutcome: No unique position for e1, combine with prev")
            mlp_input_dim = self.text_encoder.output_dim + self.event_encoder.output_dim
        else:
            mlp_input_dim = self.event_embed_dim + self.text_encoder.output_dim + self.event_encoder.output_dim

        if self.event_embeddings is not None:
            self.logits_mlp = nn.Sequential(
                    nn.Linear(mlp_input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.num_events)
                )
        else:
            logging.info("Not using Event Embeddings, Thus, Using One Hot Features for Events in Conditional Expectation Model")
            self.logits_mlp = nn.Linear(mlp_input_dim, self.num_events) #Not really a MLP, but eh


    def forward(self, input):
        """
        Params:
            (torchtext.Example) : An example from an data_utils.InstanceDataset, should contain
                .e1, Tensor [batch]
                .e2, Tensor [batch]
                .e1_text, (Tensor text [batch, max_size], Tensor lengths [batch])
        outputs:
            logits for e2 prediction, Tensor of [batch X num events]
        """

        e1_text = self.text_embeddings(input.e1_text[0]) #[batch, toklength, embd size]
        text_mask = du.create_mask(input.e1_text[0], input.e1_text[1])
        encoded_text = self.text_encoder(e1_text, mask=text_mask) #[batch, outputsize] 

        if self.includes_e1prev_intext():
            if self.onehot_events():
                e1 = self.event_encoder(input.e1.unsqueeze(dim=1), None) #batch x vocsize   #Get one hot encodings
                e1prev = self.event_encoder(input.e1prev_intext[0], None) #batch x vocsize
                events = e1 + e1prev
                events = torch.where(events < 2, events, torch.ones(events.shape, dtype=events.dtype, device=events.device)) #make sure all features are binary (not greater than 2)
                mlp_input = torch.cat([events, encoded_text], dim=1)
            elif self.combine_events:
                combined_events = torch.cat([input.e1.unsqueeze(-1), input.e1prev_intext[0]], dim=1)
                event_mask = du.create_mask(combined_events, input.e1prev_intext[1]+1)
                encoded_events = self.event_encoder(self.event_embeddings(combined_events), input.e1prev_intext[1], event_mask)
                mlp_input = torch.cat([encoded_text, encoded_events], dim=1)
            else: #Regular avg encoder
                e1 = self.event_embeddings(input.e1) #[batch, embd_size]
                event_mask = du.create_mask(input.e1prev_intext[0], input.e1prev_intext[1])
                encoded_events = self.event_encoder(self.event_embeddings(input.e1prev_intext[0]), input.e1prev_intext[1], event_mask)
                mlp_input = torch.cat([e1, encoded_text, encoded_events], dim=1)
        else:
            mlp_input = torch.cat([e1, encoded_text], dim=1)

        return self.logits_mlp(mlp_input)

    def includes_e1prev_intext(self):
        return self.event_encoder.output_dim > 0

    def onehot_events(self):
        return self.event_embeddings is None



