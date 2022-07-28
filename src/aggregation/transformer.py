"""
Transformer for encoding sequences of frame embeddings
"""
import math
import torch
import torch.nn.functional as F

from einops import rearrange, repeat
from torch import nn, einsum

# cross attention - using multi-query + one-headed key / values as in PaLM w/ optional parallel feedforward
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
    
    
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, context, attn_mask):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        # pre-layernorm, for queries and context
        x = self.norm(x)
        context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity

        sim = einsum('b h i d, b j d -> b h i j', q, k)
        # sim = sim.masked_fill(attn_mask == 0, float('-inf')) # temporarily don't use

        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if exists(self.ff):
            out = out + self.ff(x)

        return out
    
    
class AttentionalPooler(nn.Module):
    def __init__(self, dim, context_dim, seq_len, heads, dim_head, depth=1, proj_dim=None):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, dim))

        self.queries = nn.ParameterList([])
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.queries.append(
                nn.Parameter(torch.randn(seq_len, dim)),
            )
            self.layers.append(nn.ModuleList([
                CrossAttention(dim=dim, context_dim=dim, dim_head=dim_head, heads=heads, norm_context=True),
                LayerNorm(dim),
            ]))

        self.proj = None if proj_dim is None else nn.Sequential(
            nn.Linear(dim, (dim+proj_dim)//2),
            nn.GELU(),
            nn.Linear((dim+proj_dim)//2, proj_dim),
        )

    def forward(self, x, zero_masks):
        x = x.type(torch.float32)

        attn_mask = None # Temporarily don't use masking
        '''
        # create attn mask
        attn_mask = repeat(zero_masks, 'b s -> b r s', r=zero_masks.shape[-1])
        attn_mask = torch.tril(attn_mask)
        attn_mask = attn_mask.view(x.shape[0], 1, zero_masks.shape[-1], zero_masks.shape[-1])
        '''
        
        x += self.pos_encoding

        for i, (pool, norm) in enumerate(self.layers):
            img_queries = repeat(self.queries[i], 'n d -> b n d', b=x.shape[0])
            img_queries = pool(img_queries, x, attn_mask)
            img_queries = norm(img_queries)
        
        # video_embedding = img_queries[:, 0]
        video_embedding = torch.mean(img_queries, 1)
        pred = video_embedding
        if self.proj is not None:
            pred = self.proj(video_embedding)

        return pred
