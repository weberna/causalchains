################################################
#   Models for various conditional expectations
################################################
import torch
import torch.nn as nn
import numpy as np
import math


class ExpectedOutcomeNaive(nn.Module):
    'Models E[e2 | e1, e1_text]'

    def __init__(self, event_embeddings, text_embeddings, text_encoder, hidden_dim=None):
        """
        Naive expected outcome, dont adjust for previous events before e1 (M)
        Params:
            (Torch.nn.Embeddings) event_embeddings : Pytorch nn.Embeddings Module for events
            (Torch.nn.Embeddings) text_embeddings : Pytorch nn.Embeddings Module for text
            (Torch.nn.Module) text_encoder : A module for encodeing text 
                    should take in [batch X num tokens X embd dim] Tensor and output [batch X output dim vector]
        """
        super(ExpectedOutcomeNaive, self).__init__()

        self.event_embeddings = event_embeddings
        self.text_embeddings = text_embeddings
        self.text_encoder = text_encoder

        self.event_embed_dim = self.event_embeddings.weight.shape[1]
        self.text_embed_dim = self.text_embeddings.weight.shape[1]
        self.num_events = self.event_embeddings.weight.shape[0]
        self.hidden_dim = self.event_embed_dim if hidden_dim is None else hidden_dim

        mlp_input_dim = self.event_embed_dim + self.text_encoder._output_dim
        self.logits_mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.num_events)
            )


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

        e1 = self.event_embeddings(input.e1) #[batch, embd_size]
        e1_text = self.text_embeddings(input.e1_text[0]) #[batch, toklength, embd size]
#        print(e1_text)
        encoded_text = self.text_encoder(e1_text, mask=None) #[batch, outputsize], mask not needed if pad was passed to embeddings, which it is
        mlp_input = torch.cat([e1, encoded_text], dim=1)
        return self.logits_mlp(mlp_input)


        




