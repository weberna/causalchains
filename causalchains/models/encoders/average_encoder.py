########################################################################################
##   A super advanced, SOTA model for sentence encoding, Just averaging word embeddings
#   WARNING: This AI technology is so ineffably powerful that I pondered for several 
#   nights on whether to release it. Alas, I have finally given in, and decided that it 
#   would be of a greater benefit to humankind if this knowledge was known, both from a
#   practical standpoint, and from a standpoint of understanding the HUMAN MIND and 
#   the primary components of our internal language device. Indeed it was Chomsky (1965)
#   who originally propounded the idea that human language understanding it just averaging
#   together a bunch of 250 dimensional vectors together, followed by a logistic regression.
#   
#   At least I think thats what he said, I only read the back flap of Aspects, it was in Polish,
#   but I think I got the gist of it...
########################################################################################
import torch
import torch.nn as nn


#Class is pretty much here for consitancy
class AverageEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super(AverageEncoder, self).__init__()
        self._embedding_dim = self.output_dim = embedding_dim


    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor):  # pylint: disable=arguments-differ
        """
        Params:
            Tokens (Tensor[batch, maxlength, dim]) : the embeddings to avg
            lengths (Tensor[batch])
        """
        avg = torch.sum(tokens, dim=1) / lengths.view(-1, 1).type(torch.FloatTensor)
        return avg
