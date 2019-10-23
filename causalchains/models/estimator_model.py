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
from causalchains.utils.data_utils import PAD_TOK


EXP_OUTCOME_COMPONENT="_exp_outcome_component_"
PROPENSITY_COMPONENT="_propensity_component_"


class Estimator(nn.Module):
    def __init__(self, event_embed_size, text_embed_size, event_encoder_outsize, text_encoder_outsize, evocab, tvocab):
        super(Estimator, self).__init__()

        self.event_embed_size = event_embed_size
        self.text_embed_size = text_embed_size
        self.event_encoder_outsize= event_encoder_outsize
        self.text_encoder_outsize= text_encoder_outsize

        evocab_size = len(evocab.stoi.keys())
        tvocab_size = len(tvocab.stoi.keys())
        e_pad = evocab.stoi[PAD_TOK]
        t_pad = tvocab.stoi[PAD_TOK]

        self.event_embeddings = nn.Embedding(evocab_size, self.event_embed_size, padding_idx=e_pad)
        self.text_embeddings = nn.Embedding(tvocab_size, self.text_embed_size, padding_idx=t_pad)

        if self.event_encoder_outsize is not None: 
            self.event_encoder = AverageEncoder(self.event_embed_size)
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

    


class NaiveAdjustmentEstimator(Estimator):
    'Estimate ACE with Backdoor adjustment without considering any previous events'
    def __init__(self, config, evocab, tvocab):
        super(NaiveAdjustmentEstimator, self).__init__(config.event_embed_size, 
                                                       config.text_embed_size,
                                                       event_encoder_outsize=None,
                                                       text_encoder_outsize=config.text_enc_output,
                                                       evocab=evocab, tvocab=tvocab)

        self.expected_outcome = cmodels.ExpectedOutcome(self.event_embeddings, 
                                                             self.text_embeddings, 
                                                             event_encoder=None,
                                                             text_encoder=self.text_encoder,
                                                             hidden_dim=config.mlp_hidden_dim)

        assert not self.expected_outcome.includes_e1prev_intext()


    def forward(self, instance):
        exp_out = self.expected_outcome(instance)
        return {EXP_OUTCOME_COMPONENT: exp_out}
        

class SemiNaiveAdjustmentEstimator(Estimator):
    'Estimate ACE with Backdoor adjustment considering previous events that occured in text (but not those that didnt'
    def __init__(self, config, evocab, tvocab):
        super(NaiveAdjustmentEstimator, self).__init__(config.event_embed_size, 
                                                       config.text_embed_size,
                                                       event_encoder_outsize=config.event_embed_size,
                                                       text_encoder_outsize=config.text_enc_output,
                                                       evocab=evocab, tvocab=tvocab)

        self.expected_outcome = cmodels.ExpectedOutcome(self.event_embeddings, 
                                                             self.text_embeddings, 
                                                             event_encoder=self.event_encoder,
                                                             text_encoder=self.text_encoder,
                                                             hidden_dim=config.mlp_hidden_dim)


        assert self.expected_outcome.includes_e1prev_intext()

    def forward(self, instance):
        exp_out = self.expected_outcome(instance)
        return {EXP_OUTCOME_COMPONENT: exp_out}





