import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        len_idx = torch.arange(max_len)
        embed_idx = torch.arange(0, embed_dim, 2)
        embed_idx_scaled = torch.pow(10000, -embed_idx / embed_dim)
        inner = len_idx[:, None] * embed_idx_scaled[None, :] # (max_len, embed_dim / 2)
        sin_pe = torch.sin(inner) # (max_len, embed_dim / 2)
        cos_pe = torch.cos(inner)
        pe = torch.stack([sin_pe, cos_pe], dim = 2).reshape((1, max_len, embed_dim))

        # maybe easier to read alternative        
        # pe2 = torch.zeros(max_len, embed_dim)
        # pe2[:, 0::2] = torch.sin(inner)
        # pe2[:, 1::2] = torch.cos(inner)
        # pe2 = pe2.unsqueeze(0)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        output = x + self.pe[:,:S]
        output = self.dropout(output)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


def temporal_affine_backward(dout, x, W):
    N, T, D = x.shape
    No,To,M = dout.shape
    assert No == N
    assert To == T

    db = dout.sum(dim=(0,1)) # (M,)
    # (N,M,T) @ (N,T,D) = (N,M,D)
    dW = dout.transpose(1,2) @ x
    dW = torch.sum(dW, dim=0) # (M,D)
    dx = dout @ W
    return dx, dW, db

def softmax_backward(dout, probs):
    """
    dL/dp = g = dout shape (N,H,S,T)
    - derivative with respect to probs (which is output from softmax)
    p = probs shape (N,H,S,T)
    
    probs = softmax(scores)
    Goal is to take the derivative of softmax with respect to scores.
    dL/ds where s = scores

    dL/ds = J^T * dL/dp = J * dL/dp
        J is jacobian of softmax. TxT matrix
        J^T = J, since it's symmetric matrix

        J_ij = softmax(s_i)*(indicator_ij - softmax(s_j))
        J_ij = p_i * (indicator_ij - p_j)
        J_ij = p_i * indicator_ij - p_i * p_j
        J = diag(p) - p * p^T
    
    dL/ds = (diag(p) - pp^T) * g
          = diag(p)*g - pp^T * g
          = p*g      - p*(p^T * g)
          = p * (g - p^T*g)
    p^T*g is just dot product along T dimension
    
    dL/ds = p * (g - dot)
        where dot = p^T*g
    """
    
    # (N,H,S,T) .dot (N,H,S,T) = (N,H,S,1)
    dot = torch.sum(probs * dout, dim=-1, keepdims=True)
    d_scores = probs * (dout - dot)
    return d_scores


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        ############################################################################
        # TODO: Initialize any remaining layers and parameters to perform the      #
        # attention operation as defined in Transformer_Captioning.ipynb. We will  #
        # also apply dropout just after the softmax step. For reference, our       #
        # solution is less than 5 lines.                                           #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, query, key, value, attn_mask=None, return_cache = False):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the target should not be influenced by token j in the source.


        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, D = query.shape
        N, T, D = value.shape
        # Create a placeholder, to be overwritten by your code below.
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        query_embed = self.query(query) # (N,S,D)
        key_embed = self.key(key) # (N,T,D)
        value_embed = self.value(value) # (N,T,D)

        # (N,S,H,Hd) -> (N,H,S,Hd)
        query_head = query_embed.view((N,S,self.num_heads, self.head_dim)).transpose(1,2)
        # (N,T,H,Hd) -> (N,H,T,Hd) ->  (N,H,Hd,T)
        key_head = key_embed.view((N,T,self.num_heads, self.head_dim)).permute((0,2,3,1))
        value_head = value_embed.view((N,T,self.num_heads, self.head_dim)).transpose(1,2) # (N,H,T,Hd)

        scores = self.scale * (query_head @ key_head) # (N,H,S,Hd)@(N,H,Hd,T) = (N,H,S,T)
        if attn_mask is not None:
            # mask must be applied to the unnormalized scores, otherwise the scores for the mask tokens will be
            # incorporated into the denominator of softmax prob.
            scores.masked_fill_(attn_mask==0, -float('inf'))
        
        attn = torch.softmax(scores, dim=-1) # (N,H,S,T)

        # (N,H,S,T) @ (N,H,T,Hd) = (N,H,S,Hd)
        attn_out = attn @ value_head
        # attn_out = self.dropout(attn_out)
        # (N,H,S,Hd)->(N,S,H,Hd)->(N,S,D) where D=H*Hd
        attn_out = attn_out.transpose(1,2).reshape((N,S,D))
        output = self.proj(attn_out)

        if return_cache:
            cache = {
                "query": query,
                "key": key,
                "value": value,
                "query_head": query_head,
                "key_head": key_head,
                "value_head": value_head,
                "attn": attn,
                "attn_out": attn_out,
            }
            return output, cache

        return output


    def backward(self, dout, cache):
        N,S,D = dout.shape

        grads = {}
        # (N,S,D)
        d_attn_out, grads["proj.weight"], grads["proj.bias"] = temporal_affine_backward(dout, cache["attn_out"], self.proj.weight.data)
        # (N,H,S,Hd)
        d_attn_out_4d = d_attn_out.view((N,S,self.num_heads,self.head_dim)).transpose(1,2)

        # (N,H,S,Hd)x(N,H,T,Hd)
        d_attn = d_attn_out_4d @ cache["value_head"].transpose(2,3) # (N,H,S,T)
        # (N,H,S,T)x(N,H,S,Hd)
        d_value_head = cache["attn"].transpose(2,3) @ d_attn_out_4d # (N,H,T,Hd)

        T = cache["attn"].shape[-1]
        d_value_embed = d_value_head.transpose(1,2).reshape((N,T,D))
        grads["value"], grads["value.weight"], grads["value.bias"] = temporal_affine_backward(d_value_embed, cache["value"], self.value.weight.data)

        d_scores = softmax_backward(d_attn, cache["attn"]) # (N,H,S,T)

        # (N,H,S,T) @ (N,H,Hd,T) -> (N,H,S,Hd)
        d_query_head = self.scale * d_scores @ cache["key_head"].transpose(2,3)
        # (N,H,S,Hd) @ (N,H,S,T) -> (N,H,Hd,T)
        d_key_head = self.scale * cache["query_head"].transpose(2,3) @ d_scores

        d_query_embed = d_query_head.transpose(1,2).reshape((N,S,D))
        d_key_embed = d_key_head.permute((0,3,1,2)).reshape((N,T,D))

        grads["query"], grads["query.weight"], grads["query.bias"] = temporal_affine_backward(d_query_embed, cache["query"], self.query.weight.data)
        grads["key"], grads["key.weight"], grads["key.bias"] = temporal_affine_backward(d_key_embed, cache["key"], self.key.weight.data)
        return grads