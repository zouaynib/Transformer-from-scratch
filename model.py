import torch
import torch.nn as nn 
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.Embedding(x)*math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout: float ) -> None :   #d_model is the size of the positional encoding, seq_len is the length of the vector because we need a vecotr for each position and dropout is to make the model less overfit
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout

        #create a matric of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000)/d_model))
        # Apply the sin to even positions 
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply the cos to odd positions 
        pe[:, 1::2] = torch.cos(position * div_term)

        # add the batch dimension of this tensor so that we can apply it to the whole sentences ( batch of sentences ) 
        pe = pe.unsqueeze(0) #(1, seq_len, d_model)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**(-6)) -> None:
        super().__init__()
        self.eps = eps
        # eps in the denominator if sigma is very close to zero x will be very big so we add eps to make it numerically stable and to avoid division by zero
        self.alpha = nn.Parameter(torch.ones(1)) #multiplied
        self.bias = nn.Parameter(torch.zeros(0)) #added 

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True) 
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * ( x - mean )/ (std + self.eps) + self.bias

class FeedForwardlock(nn.Module) :
    def __init__(self, d_model:int, d_ff:int, dropout:float) :
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B1 
        self.d_model = d_model
        self.d_ff = d_ff
        self.droupout = nn.Dropout
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2
    def forward(self, x):
        #( Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> d_model
        return self.linear_2(self.dropout(torch.relu(self_linear_1(x))))

    
