####################################################################################
# Estimator Models
# Contains different estimators
# Each Estimator Contains:
#   -A set of required conditional distributions (nuisance parameters) that must be trained
#   -The set of parameters that are shared between nuisance parameter models (embeddings, encoders)
#   -A way to combine the estimator conditional to compute quantity of interest
####################################################################################
import torch 
import torch.nn as nn
import numpy as np
import math 
import causalchains.models.conditional_models as cmodels
from causalchains.models.encoders.cnn_encoder import CnnEncoder
from causalchains.models.encoders.average_encoder import AverageEncoder
from causalchains.models.encoders.onehot_encoder import OneHotEncoder
from causalchains.models.encoders.rnn_encoder import RnnEncoder
from causalchains.utils.data_utils import PAD_TOK
import logging


EXP_OUTCOME_COMPONENT="_exp_outcome_component_"
PROPENSITY_COMPONENT="_propensity_component_"


class Estimator(nn.Module):
    def __init__(self, event_embed_size, text_embed_size, event_encoder_outsize, text_encoder_outsize, evocab, tvocab, config):
        super(Estimator, self).__init__()

        #Pass in None for event_embed_size to just use one hot encodings for events (make sure event_encoder_outsize is the correct size)
        #Pass in None for event_encoder_outsize if not using event information

        self.event_embed_size = event_embed_size
        self.text_embed_size = text_embed_size
        self.event_encoder_outsize= event_encoder_outsize
        self.text_encoder_outsize= text_encoder_outsize

        evocab_size = len(evocab.stoi.keys())
        tvocab_size = len(tvocab.stoi.keys())
        e_pad = evocab.stoi[PAD_TOK]
        t_pad = tvocab.stoi[PAD_TOK]

        if self.event_embed_size is not None:
            self.event_embeddings = nn.Embedding(evocab_size, self.event_embed_size, padding_idx=e_pad)
        else:
            self.event_embeddings = None

        self.text_embeddings = nn.Embedding(tvocab_size, self.text_embed_size, padding_idx=t_pad)
        if config.use_pretrained:
            logging.info("Estimator: Using Pretrained Word Embeddings")
            self.text_embeddings.weight.data = tvocab.vectors

        if self.event_encoder_outsize is not None: 
            if self.event_embed_size is not None and config.rnn_event_encoder:
                logging.info("Estimator: Using RNN Event Encoder")
                self.event_encoder = RnnEncoder(self.event_embed_size, self.event_encoder_outsize)
            elif self.event_embed_size is not None:
                self.event_encoder = AverageEncoder(self.event_embed_size)
            else: 
                assert event_encoder_outsize == len(evocab.itos), "event_encoder_outsize incorrectly specified for OneHot, should be vocab size"
                self.event_encoder = OneHotEncoder(len(evocab.itos), pad_idx=e_pad)
        else:
            self.event_encoder = None

        
        if self.text_encoder_outsize is not None: 
            self.text_encoder = CnnEncoder(self.text_embed_size, 
                                             num_filters = self.text_embed_size,
                                             output_dim = self.text_encoder_outsize)
        else:
            self.text_encoder = None


    def forward(self, instance):
        """
        Params:
            Should take in a example from data_utils.InstanceDataset which will contain:
                    .e1, Tensor [batch]
                    .e2, Tensor [batch]
                    .e1_text, Tensor [batch, max_size]
                    .e1prev_intext, Tensor[batch, max_size2]
        Outputs:
            (dictonary) Output Dictionary: A dictonary mapping a component name to the logit outputs of each 
            component (for example, expected_outcome -> logits outpus of expected outcome)

        """
        raise NotImplementedError


class FineTuneEstimator(nn.Module):  #Assume using rnn encoder for events, cnn encoder for text
    'An estimator with components like the event embeddings, event encoder, text encoder,... pretrained, fixed, and passed in'
    def __init__(self, config, old_model):
        """
        Params:
            old_model (Estimator) : The previous Estimator model we are fine tunning on
        """
        super(FineTuneEstimator, self).__init__()
        self.event_embeddings = old_model.event_embeddings
        self.text_embeddings = old_model.text_embeddings
        self.event_encoder = old_model.event_encoder
        self.text_encoder = old_model.text_encoder
        self.event_embed_size = self.event_embeddings.weight.shape[1]
        self.text_embed_size = self.text_embeddings.weight.shape[1]
        self.event_encoder_outsize= self.event_encoder.output_dim
        self.text_encoder_outsize= self.text_encoder.output_dim

        self.out_event_encoder = AverageEncoder(self.event_embed_size)
        #self.out_event_encoder = OneHotEncoder(self.event_embeddings.weight.shape[0], pad_idx=self.event_embeddings.padding_idx)
        
    


class NaiveAdjustmentEstimator(Estimator):
    'Estimate ACE with Backdoor adjustment without considering any previous events'
    def __init__(self, config, evocab, tvocab):
        super(NaiveAdjustmentEstimator, self).__init__(config.event_embed_size, 
                                                       config.text_embed_size,
                                                       event_encoder_outsize=None,
                                                       text_encoder_outsize=config.text_enc_output,
                                                       evocab=evocab, tvocab=tvocab, config=config)

        self.expected_outcome = cmodels.ExpectedOutcome(self.event_embeddings, 
                                                             self.text_embeddings, 
                                                             event_encoder=None,
                                                             text_encoder=self.text_encoder,
                                                             evocab=evocab,tvocab=tvocab,
                                                             config=config)

        assert not self.expected_outcome.includes_e1prev_intext()
        assert not self.expected_outcome.onehot_events()

    def forward(self, instance):
        exp_out = self.expected_outcome(instance)
        return {EXP_OUTCOME_COMPONENT: exp_out}
        

class AdjustmentEstimator(FineTuneEstimator):
    'Estimate ACE with Backdoor adjustment considering previous events that occured in and out of text, used with finetunning'
    def __init__(self, config, evocab, tvocab, old_model):
        super(AdjustmentEstimator, self).__init__(config, old_model)

        mlp_layer = old_model.expected_outcome.logits_mlp #reuse part of previous last layer
        self.expected_outcome = cmodels.ExpectedOutcome(self.event_embeddings, 
                                                             self.text_embeddings, 
                                                             event_encoder=self.event_encoder,
                                                             text_encoder=self.text_encoder,
                                                             evocab=evocab,tvocab=tvocab,
                                                             config=config,out_event_encoder=self.out_event_encoder, old_mlp_layer=mlp_layer)


        assert self.expected_outcome.includes_e1prev_intext()

    def forward(self, instance):
        exp_out = self.expected_outcome(instance)
        return {EXP_OUTCOME_COMPONENT: exp_out}


class SemiNaiveAdjustmentEstimator(Estimator):
    'Estimate ACE with Backdoor adjustment considering previous events that occured in text (but not those that didnt appear in text)'
    def __init__(self, config, evocab, tvocab):
        super(SemiNaiveAdjustmentEstimator, self).__init__(config.event_embed_size, 
                                                       config.text_embed_size,
                                                       event_encoder_outsize=config.rnn_hidden_dim, #assume using rnn event encoder
                                                       text_encoder_outsize=config.text_enc_output,
                                                       evocab=evocab, tvocab=tvocab, config=config)

        self.expected_outcome = cmodels.ExpectedOutcome(self.event_embeddings, 
                                                             self.text_embeddings, 
                                                             event_encoder=self.event_encoder,
                                                             text_encoder=self.text_encoder,
                                                             evocab=evocab,tvocab=tvocab,
                                                             config=config)


        assert self.expected_outcome.includes_e1prev_intext()
        assert not self.expected_outcome.onehot_events()

    def forward(self, instance):
        exp_out = self.expected_outcome(instance)
        return {EXP_OUTCOME_COMPONENT: exp_out}

class SemiNaiveAdjustmentEstimatorOneHotEvents(Estimator):
    'Estimate ACE with Backdoor adjustment considering previous events that occured in text (but not those that didnt appear in text)'
    def __init__(self, config, evocab, tvocab):
        super(SemiNaiveAdjustmentEstimatorOneHotEvents, self).__init__(None, 
                                                       config.text_embed_size,
                                                       event_encoder_outsize=len(evocab.itos),
                                                       text_encoder_outsize=config.text_enc_output,
                                                       evocab=evocab, tvocab=tvocab, config=config)

        self.expected_outcome = cmodels.ExpectedOutcome(self.event_embeddings, 
                                                             self.text_embeddings, 
                                                             event_encoder=self.event_encoder,
                                                             text_encoder=self.text_encoder,
                                                             evocab=evocab,tvocab=tvocab,
                                                             config=config)


        assert self.expected_outcome.includes_e1prev_intext()
        assert self.expected_outcome.onehot_events()

    def forward(self, instance):
        exp_out = self.expected_outcome(instance)
        return {EXP_OUTCOME_COMPONENT: exp_out}




