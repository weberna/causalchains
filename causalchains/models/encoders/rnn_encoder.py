import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


#Class is pretty much here for consitancy
class RnnEncoder(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int) -> None:
        super(RnnEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self.output_dim = self.hidden_dim = output_dim
        self.rnn = nn.GRU(self._embedding_dim, self.hidden_dim, 1, bidirectional=False, batch_first=True)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor):  # pylint: disable=arguments-differ
        """
        Params:
            Tokens (Tensor[batch, maxlength, dim]) : the input embeddings
            lengths (Tensor[batch])
        Outputs:
            encoded states (Tensor[batch, output_size])
        """
        packed_input = pack_padded_sequence(tokens, lengths.cpu().numpy(), batch_first=True)
        self.rnn.flatten_parameters()
        _, last_state = self.rnn(packed_input) #[1, batch, hiddensize]
        last_state=last_state.squeeze(dim=0)
        return last_state

