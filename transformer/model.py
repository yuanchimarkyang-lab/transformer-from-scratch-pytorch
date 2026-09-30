import torch
import numpy as np
import math
from torchinfo import summary


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, Q, K, V, Q_mask, K_mask, causal_mask=None):
        # Q: torch.tensor of dimension Batch X #_Qs X D_Q (D_Q == D_K)
        # K: torch.tensor of dimension Batch X #_Ks X D_K
        # V: torch.tensor of dimension Batch X #_Vs X D_V
        # x: torch.tensor of dimension Batch X 

        # K_mask: torch.tensor of dimention (Batch, 1, #_Ks)
        Q_mask = torch.transpose(Q_mask, 1,2) # convert to (Batch, #_Qs, 1)
        D_K = K.shape[1]

        x = torch.bmm(Q,torch.transpose(K,1,2))/np.sqrt(D_K) # (Batch, #_Qs, #_Ks)
        
        if causal_mask is not None:
            x = x.masked_fill(~causal_mask, float("-inf"))


        x = x.masked_fill(K_mask,float("-inf"))

        score = self.softmax(x)
        x = torch.bmm(score,V) # (Batch, #_Qs, D_V) with assumption that #_Ks == #_Vs
        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, D_model=512, h=8, D_V=64):
        super().__init__()
        self.D_model = D_model
        self.h = h
        self.D_K = D_model//h
        self.D_V = D_V
        self.W_Ks = torch.nn.ModuleList([torch.nn.Linear(self.D_model, self.D_K) for _ in range(h)])
        self.W_Qs = torch.nn.ModuleList([torch.nn.Linear(self.D_model, self.D_K) for _ in range(h)])
        self.W_Vs = torch.nn.ModuleList([torch.nn.Linear(self.D_model, self.D_V) for _ in range(h)])
        self.SDPAs = torch.nn.ModuleList([ScaledDotProductAttention() for _ in range(h)])

        self.W_O = torch.nn.Linear(self.h*self.D_V, self.D_model)
        

    def forward(self, Q, K, V, Q_mask, K_mask, causal_mask=None):
        heads = []
        for i in range(self.h):
            Qi = self.W_Qs[i](Q)
            Ki = self.W_Ks[i](K)
            Vi = self.W_Vs[i](V)
            heads.append(self.SDPAs[i](Qi,Ki,Vi, Q_mask, K_mask, causal_mask))
            
        x = torch.cat(heads, dim=2) # (Batch, #_Qs, h*D_V)
        x = self.W_O(x) # (Batch, #_Qs, D_model)
        return x


class EncoderBlock(torch.nn.Module):
    def __init__(self, D_model=512, h=8, D_V=64, D_FFN = 2048,p=0.1):
        super().__init__()
        self.D_model = D_model
        self.h = h
        self.D_K = D_model//h
        self.D_V = D_V

        self.LayerNorm1 = torch.nn.LayerNorm(self.D_model)
        self.LayerNorm2 = torch.nn.LayerNorm(self.D_model)
        
        self.FeedForward1 = torch.nn.Linear(D_model, D_FFN)
        self.ReLU = torch.nn.ReLU()
        self.FeedForward2 = torch.nn.Linear(D_FFN, D_model)
        
        self.MLA = MultiHeadAttention(D_model=D_model, h=h, D_V=D_V)
        self.dropout = torch.nn.Dropout(p=p)

    def forward(self, x, x_mask):
        y = self.MLA(x,x,x,x_mask,x_mask, causal_mask=None)
        y = self.dropout(y)
        x = self.LayerNorm1(x+y)
        y = self.FeedForward1(x)
        y = self.ReLU(y)
        y = self.FeedForward2(y)
        y = self.dropout(y)
        x = self.LayerNorm2(x+y)
        return x


class DecoderBlock(torch.nn.Module):
    
    def __init__(self, D_model=512, h=8, D_V=64, D_FFN = 2048,p=0.1):
        super().__init__()
        self.D_model = D_model
        self.h = h
        self.D_K = D_model//h
        self.D_V = D_V
               

        self.LayerNorm1 = torch.nn.LayerNorm(D_model)
        self.LayerNorm2 = torch.nn.LayerNorm(D_model)
        self.LayerNorm3 = torch.nn.LayerNorm(D_model)
        self.FeedForward1 = torch.nn.Linear(D_model, D_FFN)
        self.ReLU = torch.nn.ReLU()
        self.FeedForward2 = torch.nn.Linear(D_FFN, D_model)
        self.MMLA = MultiHeadAttention(D_model=D_model, h=h, D_V=D_V)
        self.CA = MultiHeadAttention(D_model=D_model, h=h, D_V=D_V)
        self.dropout = torch.nn.Dropout(p=p)

    def forward(self, x, x_encoder, x_mask, x_encoder_mask):
        causal_mask = torch.tril(torch.ones(x.shape[1], x.shape[1]), diagonal=0).bool().to(x.device)
        y = self.MMLA(x,x,x, x_mask, x_mask, causal_mask)
        y = self.dropout(y)
        x = self.LayerNorm1(x+y)

        causal_mask = None
        y = self.CA(x,x_encoder,x_encoder,x_mask, x_encoder_mask, causal_mask)
        y = self.dropout(y)
        x = self.LayerNorm2(x+y)
        
        y = self.FeedForward1(x)
        y = self.ReLU(y)
        y = self.FeedForward2(y)
        
        y = self.dropout(y)
        x = self.LayerNorm3(x+y)
        return x

class PositionalEncoding(torch.nn.Module):
    def __init__(self, embedding_dim, max_len):
        super().__init__()
        # calculating the parameters for the cosine/sine function
        pos_array = np.expand_dims(np.array(range(max_len)),axis=1)
        i_array = np.expand_dims(np.array([1/np.power(10000.0, 2*i/embedding_dim) for i in range(embedding_dim//2+embedding_dim%2)]),axis=0)
        
        pos_i_matrix = torch.from_numpy(np.dot(pos_array,i_array)).unsqueeze(0)
        
        
        # calculating the Positional Encoding
        PE = torch.zeros((1, max_len, embedding_dim))
        PE[0, :,0::2] = torch.sin(pos_i_matrix)
        if embedding_dim % 2 == 0:
            PE[0, :,1::2] = torch.cos(pos_i_matrix)
        else:
            PE[0, :,1::2] = torch.cos(pos_i_matrix[0,:,:-1])
        self.register_buffer('pe', PE) # self.pe is of dimension (max_len, embedding_dim)
        

    def forward(self, x):
        """
            x: batch array with size (batch_size, seq_length, embedding_dim)
        """
        x = x + self.pe[:, :x.shape[1], :]

        return x


class transformer(torch.nn.Module):
    def __init__(self, D_model=512, h=8, Dict_size = 8000, N_encoder = 6, N_decoder = 6, D_FFN=2048, p=0.1, 
                       PAD=0, max_len=100):
        super().__init__()
        self.D_model = D_model
        self.h = h
        assert D_model%h == 0, "D_model % h is not 0!"
        self.D_K = D_model//h
        self.D_V = D_model//h
        self.N_encoder = N_encoder
        self.N_decoder = N_decoder
        self.PAD = PAD
        self.Dict_size = Dict_size
        self.max_len = max_len
        self.D_FFN = D_FFN
        self.p = p

        self.InputEmbeddings = torch.nn.Embedding(self.Dict_size, self.D_model, padding_idx=self.PAD) # padding = 0
        self.InputPE = PositionalEncoding(embedding_dim = self.D_model, max_len = self.max_len)
        self.OutputEmbeddings = torch.nn.Embedding(self.Dict_size, self.D_model, padding_idx=self.PAD) # padding = 0
        self.OutputPE = PositionalEncoding(embedding_dim = self.D_model, max_len = self.max_len)

        self.EncoderBlocks = torch.nn.ModuleList([
            EncoderBlock(D_model=self.D_model, h=self.h, D_V=self.D_V, D_FFN=self.D_FFN, p=self.p) 
            for _ in range(N_encoder)])
        self.DecoderBlocks = torch.nn.ModuleList([
            DecoderBlock(D_model=self.D_model, h=self.h, D_V=self.D_V, D_FFN=self.D_FFN, p=self.p) 
            for _ in range(N_decoder)])

        self.Linear = torch.nn.Linear(self.D_model,self.Dict_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(p=self.p)

    def forward(self, encoder_input, decoder_input):
        encoder_input_mask = (encoder_input==self.PAD).unsqueeze(1)
        decoder_input_mask = (decoder_input==self.PAD).unsqueeze(1)
        
        encoder_output = self.InputEmbeddings(encoder_input) # (batch, #_input, D_model)
        encoder_output = self.InputPE(encoder_output)               # (batch, #_input, D_model)
        encoder_output = self.dropout(encoder_output)
        for i in range(self.N_encoder):
            encoder_output = self.EncoderBlocks[i](encoder_output, encoder_input_mask)  # (batch, #_input, D_model)

        decoder_output = self.OutputEmbeddings(decoder_input) # (batch, #_output, D_model)
        decoder_output = self.OutputPE(decoder_output)        # (batch, #_output, D_model)
        decoder_output = self.dropout(decoder_output)
        for i in range(self.N_decoder):
            decoder_output = self.DecoderBlocks[i](decoder_output, encoder_output, decoder_input_mask, encoder_input_mask) # (batch, #_output, D_model)

        decoder_output = self.Linear(decoder_output) # (batch, #_output, dict_size)
        decoder_output = torch.transpose(decoder_output,1,2) # (batch, dict_size, #_output)

        return decoder_output # logits

if __name__ == "__main__":
    PE_model = PositionalEncoding(embedding_dim = 10, max_len = 10)
    x = torch.zeros((2,6,10))
    y = PE_model(x)
    print("x = ", x)
    print("PE_model(x) = ", y)