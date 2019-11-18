import torch
import torch.nn as nn

class EventLM(nn.Module): 

    def __init__(self, ninput, nhidden, nlayers, nvocab, vocab=None, rnn_type="GRU", dropout=0.1):

        super(EventLM, self).__init__()

        self.dropout = nn.Dropout(dropout)
        # word embedding layer
        self.embedding = nn.Embedding(nvocab, ninput)

        self.rnn = nn.GRU(ninput, nhidden, nlayers, batch_first=True)
        # logit layer
        self.linear_out = nn.Linear(nhidden, nvocab)
        
        self.rnn_type = rnn_type
        self.nhidden = nhidden
        self.nlayers = nlayers
        self.vocab = vocab


    def forward(self, input, hidden, input2=None): 
        """
        dropout is on the input and output

        input is a [batch, 1] size Tensor 
        """
        batch_size = input.size(0)
        # word embedding [batch X 1 X emb_dim]
        emb = self.dropout(self.embedding(input))

        output, hidden = self.rnn(emb, hidden)

        # output [batch_size, 1, hidden_size]
        output = self.dropout(output)
        # logit is [batch_size , vocab]
        logit = self.linear_out(output.view(output.size(0)*output.size(1), output.size(2))) 
        return logit, hidden

