import torch
import torch.nn as nn 
import math

class InputEmbeddings(nn.module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.Embedding(x)*math.sqrt(self.d_model)


class PositionalEncoding(nn.module):
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

    