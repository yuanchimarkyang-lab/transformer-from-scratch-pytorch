"""
    This script implements an encoder-decoder architecture from scratch. The components implements including
        - scaled dot-product attention
        - multi-head attention
        - encoder/decoder
        - positional encoding
        - causal and padding masks

"""

import torch
import numpy as np
import math
from torchinfo import summary


class ScaledDotProductAttention(torch.nn.Module):
    """
        This class implements Scaled-Dot-Product Attention
    """
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, Q, K, V, K_mask, causal_mask=None):
        """
            Parameters: 
                Q: torch.tensor of (batch_size, Seq_length_Q, D_Q)
                K: torch.tensor of (batch_size, Seq_length_K, D_K)
                V: torch.tensor of (batch_size, Seq_length_V, D_V)
                K_mask: the padding mask of size (Seq_length_K,Seq_length_K)
                causal_mask: the causal mask of size (Seq_length_Q,Seq_length_Q)
           
            Returns:
                x: torch.tensor of (batch_size, Seq_length_Q, D_V)

            Note: here we follow the paper and make D_Q == D_K and Seq_length_K == Seq_length_V
        
        """

        D_K = K.shape[1]

        x = torch.bmm(Q,torch.transpose(K,1,2))/np.sqrt(D_K) # (Batch, #_Qs, #_Ks)
        
        if causal_mask is not None:
            x = x.masked_fill(~causal_mask, float("-inf"))


        x = x.masked_fill(K_mask,float("-inf"))

        score = self.softmax(x)
        x = torch.bmm(score,V) # (Batch, #_Qs, D_V) with assumption that Seq_length_K == Seq_length_V
        return x


class MultiHeadAttention(torch.nn.Module):
    """
        This class implements Multi-Head Attention Head
    """
    def __init__(self, D_model=512, h=8):
        """
            Parameters:
                D_model: the embedding_dim
                h: the attention heads
        """
        super().__init__()
        self.D_model = D_model
        self.h = h
        self.D_K = D_model//h
        self.D_V = D_model//h
        self.W_Ks = torch.nn.ModuleList([torch.nn.Linear(self.D_model, self.D_K) for _ in range(h)])
        self.W_Qs = torch.nn.ModuleList([torch.nn.Linear(self.D_model, self.D_K) for _ in range(h)])
        self.W_Vs = torch.nn.ModuleList([torch.nn.Linear(self.D_model, self.D_V) for _ in range(h)])
        self.SDPAs = torch.nn.ModuleList([ScaledDotProductAttention() for _ in range(h)])
        self.W_O = torch.nn.Linear(self.h*self.D_V, self.D_model)
        

    def forward(self, Q, K, V, K_mask, causal_mask=None):
        """
            Parameters: 
                Q: torch.tensor of (batch_size, Seq_length_Q, D_model)
                K: torch.tensor of (batch_size, Seq_length_K, D_model)
                V: torch.tensor of (batch_size, Seq_length_V, D_model)
                K_mask: the padding mask of size (Seq_length_K,Seq_length_K)
                causal_mask: the causal mask of size (Seq_length_Q,Seq_length_Q)
           
            Returns:
                x: torch.tensor of (batch_size, Seq_length_Q, D_model)

            Note: here we follow the paper and make D_Q == D_K and Seq_length_K == Seq_length_V
        
        """
        heads = []
        for i in range(self.h):
            Qi = self.W_Qs[i](Q)
            Ki = self.W_Ks[i](K)
            Vi = self.W_Vs[i](V)
            heads.append(self.SDPAs[i](Qi,Ki,Vi, K_mask, causal_mask))
            
        x = torch.cat(heads, dim=2) # (batch_size, Seq_length_Q, h*D_V)
        x = self.W_O(x) # (batch_size, Seq_length_Q, D_model)
        return x


class EncoderBlock(torch.nn.Module):
    """
        This class implements the Encoder Block
    """
    def __init__(self, D_model=512, h=8, D_FFN = 2048,p=0.1):
        """
            Parameters:
                D_model: the embedding_dim
                h: the attention heads
                D_FFN: the hidden size of the Position-Wise Feed-Forward Network
                p: the dropout probability
        """
        super().__init__()
        self.D_model = D_model
        self.h = h

        self.LayerNorm1 = torch.nn.LayerNorm(self.D_model)
        self.LayerNorm2 = torch.nn.LayerNorm(self.D_model)
        
        self.FeedForward1 = torch.nn.Linear(D_model, D_FFN)
        self.ReLU = torch.nn.ReLU()
        self.FeedForward2 = torch.nn.Linear(D_FFN, D_model)
        
        self.MLA = MultiHeadAttention(D_model=D_model, h=h)
        self.dropout = torch.nn.Dropout(p=p)

    def forward(self, x, x_mask):
        """
            Parameters: 
                x: torch.tensor of (batch_size, Seq_length_Q, D_model) as Q, K, and V
                x_mask: the padding mask of size (Seq_length_K,Seq_length_K) as K_mask
           
            Returns:
                x: torch.tensor of (batch_size, Seq_length_Q, D_model)
        
        """
        # the multi-head attention
        y = self.MLA(Q=x,K=x,V=x,K_mask=x_mask,causal_mask=None)
        # the droptout after the output of the sublayer
        y = self.dropout(y)
        # Add & Norm
        x = self.LayerNorm1(x+y)

        # the Position-wise Feed-Forward Network
        y = self.FeedForward1(x)
        y = self.ReLU(y)
        y = self.FeedForward2(y)
        # the droptout after the output of the sublayer
        y = self.dropout(y)
        # Add & Norm
        x = self.LayerNorm2(x+y)
        
        return x


class DecoderBlock(torch.nn.Module):
    """
        This class implements the decoder Block
    """    
    def __init__(self, D_model=512, h=8, D_FFN = 2048,p=0.1):
        super().__init__()
        """
            Parameters:
                D_model: the embedding_dim
                h: the attention heads
                D_FFN: the hidden size of the Position-Wise Feed-Forward Network
                p: the dropout probability
        """ 
        self.D_model = D_model
        self.h = h
        self.D_K = D_model//h
               

        self.LayerNorm1 = torch.nn.LayerNorm(D_model)
        self.LayerNorm2 = torch.nn.LayerNorm(D_model)
        self.LayerNorm3 = torch.nn.LayerNorm(D_model)
        self.FeedForward1 = torch.nn.Linear(D_model, D_FFN)
        self.ReLU = torch.nn.ReLU()
        self.FeedForward2 = torch.nn.Linear(D_FFN, D_model)
        self.MMLA = MultiHeadAttention(D_model=D_model, h=h)
        self.CA = MultiHeadAttention(D_model=D_model, h=h)
        self.dropout = torch.nn.Dropout(p=p)

    def forward(self, x, x_encoder, x_mask, x_encoder_mask):
        """
            Parameters: 
                x: torch.tensor of (batch_size, Seq_length_Q, D_model) as Q, K, and V for the self-attention and 
                                                                          Q           for cross-attention
                x_encoder: torch.tensor of (batch_size, Seq_length_Q, D_model) as K and V for cross-attention
                x_mask: the padding mask of size (Seq_length_K,Seq_length_K) as K_mask for self-atention
                x_encoder_mask: the padding mask of size (Seq_length_K,Seq_length_K) as K_mask for cross-attention
           
            Returns:
                x: torch.tensor of (batch_size, Seq_length_Q, D_model)
        
        """ 
        # createing causal_mask as an lower triangular matrix of shape (Seq_length_Q,Seq_length_Q)
        causal_mask = torch.tril(torch.ones(x.shape[1], x.shape[1]), diagonal=0).bool().to(x.device)
        
        # the Masked Multi-Head Attention
        y = self.MMLA(Q=x,K=x,V=x, K_mask = x_mask, causal_mask=causal_mask)
        # the droptout after the output of the sublayer
        y = self.dropout(y)
        # Add & Norm
        x = self.LayerNorm1(x+y)

        # setting causal_mask as None
        causal_mask = None

        # the Cross Attention
        y = self.CA(Q=x,K=x_encoder,V=x_encoder,K_mask=x_encoder_mask, causal_mask=causal_mask)
        # the droptout after the output of the sublayer
        y = self.dropout(y)
        # Add & Norm
        x = self.LayerNorm2(x+y)

        # the Position-wise Feed-Forward Network
        y = self.FeedForward1(x)
        y = self.ReLU(y)
        y = self.FeedForward2(y)
        # the droptout after the output of the sublayer
        y = self.dropout(y)
        # Add & Norm
        x = self.LayerNorm3(x+y)
        return x

class PositionalEncoding(torch.nn.Module):
    """
        This class implements the Positional Encoding
    """    
    def __init__(self, embedding_dim, max_len):
        """
            Parameters:
                embedding_dim: the dimention of the embedding
                max_len: the maximum sequence length 
        """ 
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
            Parameters
                x: torch array with size (batch_size, seq_length, embedding_dim)

            Return:
                x: torch array with size (batch_size, seq_length, embedding_dim)
        """
        x = x + self.pe[:, :x.shape[1], :]

        return x


class transformer(torch.nn.Module):
    """
        This class implements an encoder-decoder transformer architecture
    """   
    def __init__(self, D_model=512, h=8, Vocab_size = 8000, N_encoder = 6, N_decoder = 6, D_FFN=2048, p=0.1, 
                       PAD=0, max_len=100):
        """
            Parameters:
                D_model: the embedding_dim
                h: the attention heads
                Vocab_size: the vocabulary size
                N_encoder: the number of encoder block
                N_decoder: the number of decoder block
                D_FFN: the hidden size of the Position-Wise Feed-Forward Network
                p: the dropout probability
                PAD: the PAD ID
                max_len: maximum sequence length

        """ 
        super().__init__()
        self.D_model = D_model
        self.h = h
        assert D_model%h == 0, "D_model % h is not 0!"
        self.N_encoder = N_encoder
        self.N_decoder = N_decoder
        self.PAD = PAD
        self.Vocab_size = Vocab_size
        self.max_len = max_len
        self.D_FFN = D_FFN
        self.p = p

        self.InputEmbeddings = torch.nn.Embedding(self.Vocab_size, self.D_model, padding_idx=self.PAD) # padding = 0
        self.InputPE = PositionalEncoding(embedding_dim = self.D_model, max_len = self.max_len)
        self.OutputEmbeddings = torch.nn.Embedding(self.Vocab_size, self.D_model, padding_idx=self.PAD) # padding = 0
        self.OutputPE = PositionalEncoding(embedding_dim = self.D_model, max_len = self.max_len)

        self.EncoderBlocks = torch.nn.ModuleList([
            EncoderBlock(D_model=self.D_model, h=self.h, D_FFN=self.D_FFN, p=self.p) 
            for _ in range(N_encoder)])
        self.DecoderBlocks = torch.nn.ModuleList([
            DecoderBlock(D_model=self.D_model, h=self.h, D_FFN=self.D_FFN, p=self.p) 
            for _ in range(N_decoder)])

        self.Linear = torch.nn.Linear(self.D_model,self.Vocab_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(p=self.p)

    def forward(self, encoder_input, decoder_input):
        """
            Parameters:
                encoder_input: torch array with size (batch_size, seq_length)
                decoder_input: torch array with size (batch_size, seq_length)
            Return:
                decoder_output: torch array with shape (batch_size, Vocab_size, seq_length) as logits
        """
        # Creating padding mask
        encoder_input_mask = (encoder_input==self.PAD).unsqueeze(1)
        decoder_input_mask = (decoder_input==self.PAD).unsqueeze(1)
        
        # Apply Embedding and Positional Encoding to the encoder input
        encoder_output = self.InputEmbeddings(encoder_input) # (batch_size, seq_length, D_model)
        encoder_output = self.InputPE(encoder_output)        # (batch_size, seq_length, D_model)
        # Dropout
        encoder_output = self.dropout(encoder_output)
        # EncoderBlocks
        for i in range(self.N_encoder):
            encoder_output = self.EncoderBlocks[i](encoder_output, encoder_input_mask)  # (batch_size, seq_length, D_model)

        # Apply Embedding and Positional Encoding to the decoder input
        decoder_output = self.OutputEmbeddings(decoder_input) # (batch_size, seq_length, D_model)
        decoder_output = self.OutputPE(decoder_output)        # (batch_size, seq_length, D_model)
        # Dropout
        decoder_output = self.dropout(decoder_output)
        # DecoderBlocks
        for i in range(self.N_decoder):
            decoder_output = self.DecoderBlocks[i](decoder_output, encoder_output, decoder_input_mask, encoder_input_mask) # (batch, #_output, D_model)
        # Linear layer
        decoder_output = self.Linear(decoder_output) # (batch_size, seq_length, Vocab_size)
        # Transpose for logits
        decoder_output = torch.transpose(decoder_output,1,2) # (batch_size, seq_length, #_output)

        return decoder_output 
