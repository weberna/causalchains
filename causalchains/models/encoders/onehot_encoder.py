import torch
import torch.nn as nn


#Class is pretty much here for consitancy
class OneHotEncoder(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int) -> None:
        super(OneHotEncoder, self).__init__()
        self._embedding_dim = self.output_dim = vocab_size
        self.pad_idx = pad_idx


    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor):  # pylint: disable=arguments-differ
        """
        Params:
            Tokens (Tensor[batch, maxlength]) : the token ids 
            lengths (Tensor[batch])
        returns:
            Tensor[batch, vocab dim]
            
        """

        tokens = tokens.unsqueeze(dim=2)
        pad_selector = torch.zeros(tokens.shape, dtype=tokens.dtype, device=tokens.device).fill_(self.pad_idx)
        onehot = torch.zeros(tokens.shape[0], tokens.shape[1], self.output_dim, device=tokens.device).scatter(2, tokens, 1) #[batch, maxlen, vocabsize]
        onehot = onehot.scatter(2, pad_selector, 0) #zero out the pad element
        onehot = onehot.sum(dim=1) #[batch, vocabsize]
        outhot = torch.where(onehot < 2, onehot, torch.ones(onehot.shape, device=onehot.device)) #make sure all features are binary (not greater than 2)

        return onehot
