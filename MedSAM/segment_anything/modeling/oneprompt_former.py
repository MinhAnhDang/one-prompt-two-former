import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing import List, Tuple, Type
from .common import LayerNorm2d
from einops import rearrange
import math
from .image_encoder import PatchEmbed
from .transformer import TwoWayTransformer, Attention, TwoWayAttentionBlock, CrossAttentionBlock
import math
from .common import MLPBlock
# from functools import partial
from einops.layers.torch import Rearrange, Reduce
import numpy as np
import gc


pair = lambda x: x if isinstance(x, tuple) else (x, x)

def gaussian_kernel(size, mean, std):
    """Generates a 2D Gaussian kernel."""
    d = torch.distributions.Normal(mean, std)
    vals = d.log_prob(torch.arange(size).float())
    grid = torch.exp(vals[:, None] + vals[None, :])
    grid /= grid.sum()
    return grid

class GaussianConv2d(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, kernel_size = 3, stride=1, padding=1, mean=0.0, std=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mean = nn.Parameter(torch.tensor(mean), requires_grad=True)
        self.std = nn.Parameter(torch.tensor(std), requires_grad=True)
        self.weights = nn.Parameter(gaussian_kernel(kernel_size, self.mean, self.std), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)

    def forward(self, x):
        return F.conv2d(x, self.weights.unsqueeze(0).unsqueeze(0).repeat(self.out_channels, self.in_channels, 1, 1),
                        bias=self.bias, stride=self.stride, padding=self.padding)


def PromptMLP(dim = 3, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, 1),
        nn.Dropout(dropout)
    )


class PromptMixer(nn.Module):
    def __init__(
        self,
        dim: int = 3,
        depth: int = 1,
        expansion_factor: int = 4,
        dropout: float = 0.,
    ) -> None:
        
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.layers = nn.Sequential(
        Rearrange('k b n d -> b n d k'),
        *[nn.Sequential(
            PromptMLP(dim, expansion_factor, dropout),
        ) for _ in range(depth)],
        # nn.LayerNorm(dim) # b n d
    )

    def forward(self, q, k, v):
        qk = torch.stack([q, k, v]) # 3 b n d
        res = self.layers(qk)
        # print("res size is", res.size())
        return res.squeeze(-1) # b n d


class PromptParser(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        token_num: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.pt_mix = PromptMixer()
        self.gauss = GaussianConv2d(in_channels = token_num)

        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        tmp_embedding: Tensor,
        prompt_embedding1: Tensor,
        prompt_embedding2: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        # pt_pe = prompt_embedding1 + prompt_embedding2
        etpp = self.pt_mix(tmp_embedding, prompt_embedding1, prompt_embedding2)
        # print("etpp shape is", etpp.size())
        # print("image_embedding shape is", image_embedding.size())
        b, n, c = etpp.size()
        b, n, x = image_embedding.size()
        att_m = torch.matmul(etpp.view(-1, c).unsqueeze(-1), image_embedding.view(-1, x).unsqueeze(-2)).view(b, n, c, x)
        # att_m = torch.einsum ('bncd, bndx -> bncx', etpp.unsqueeze(-1), image_embedding.unsqueeze(-2)) 
        att_m = self.gauss(att_m)
        etq = torch.matmul(image_embedding.view(-1, x).unsqueeze(-1), (tmp_embedding + prompt_embedding1 + prompt_embedding2).view(-1, c).unsqueeze(-2)).view(b, n, x, c)
        # etq = torch.einsum ('bncd, bndx -> bncx', image_embedding.unsqueeze(-1), (tmp_embedding + pt_pe).unsqueeze(-2))
        att_m = torch.max(torch.matmul(att_m, etq), etq)
        # print("Memory allocated", torch.cuda.memory_allocated(0)/1e9)
        # print("Reversed memory", torch.cuda.memory_reserved(0)/1e9)
        # att_m = torch.matmul(att_m, etq)
        # res = torch.einsum ('bncx, bnx -> bnc', att_m, tmp_embedding + prompt_embedding1 + prompt_embedding2) 
        res =  torch.matmul(att_m, (tmp_embedding + prompt_embedding1 + prompt_embedding2).unsqueeze(-1)).squeeze(-1)
        gc.collect()
        torch.cuda.empty_cache()
        return image_embedding, res


class _OnePromptFormer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        prompt_embed_dim: int,
        token_num: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        self.layers = nn.ModuleList()

        self.nn = nn.Linear(embedding_dim, prompt_embed_dim)

        self.attns1 = Attention(prompt_embed_dim, num_heads)
        self.attns2 = Attention(prompt_embed_dim, num_heads)
        self.mlps1 = MLPBlock(prompt_embed_dim, mlp_dim, activation)
        self.norms1 = nn.LayerNorm(prompt_embed_dim)
        self.norms2 = nn.LayerNorm(prompt_embed_dim)


        self.parser = PromptParser(embedding_dim = prompt_embed_dim, token_num = token_num)
        self.attnt1 = Attention(prompt_embed_dim, num_heads)
        self.mlpt1 = MLPBlock(prompt_embed_dim, mlp_dim, activation)
        self.normt1 = nn.LayerNorm(prompt_embed_dim)
        self.normt2 = nn.LayerNorm(prompt_embed_dim)

        self.attnm1 = Attention(prompt_embed_dim, num_heads)
        self.attnm2 = Attention(prompt_embed_dim, num_heads)

        self.final = nn.Sequential(
            MLPBlock(prompt_embed_dim, mlp_dim, activation),
            nn.LayerNorm(prompt_embed_dim)
        )

    def forward(
        self,
        emb: Tensor,
        image_embedding: Tensor,
        tmp_embedding: Tensor,
        prompt_embedding1: Tensor,
        prompt_embedding2: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        image_embedding, et = self.parser(image_embedding,tmp_embedding, prompt_embedding1, prompt_embedding2)
        gc.collect()
        torch.cuda.empty_cache()
        # print("et size is", et.size())
        # print("image_embedding size is", image_embedding.size())
        es = self.attns1(q=image_embedding, k= emb, v= emb)
        es_bk = es
        es = self.attns2(q=et, k= es, v= es)
        es = self.norms1(es + et)
        es = self.norms2(self.mlps1(es) + es)

        et = self.attnt1(q = es_bk, k = et, v = et)
        et = self.normt1(es_bk + et)
        et = self.norms2(self.mlps1(et) + et)

        e = self.attnm1(q = et, k = es, v = es)
        e = self.attnm2(q = e, k = e, v = e)
        e = self.final(e)

        return e


class MixedUpScale(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                CrossAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys
    
    
class Decode_Align(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        transformer_dim: int,
        stages: int = 4096,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim

        self.num_mask_tokens = stages
        # print("num_mask_tokens", self.num_mask_tokens)
        self.p1_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.p2_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.layer = nn.Linear(embed_dim, transformer_dim)

    def forward(
        self,
        x:torch.Tensor,
        src_embeddings:torch.Tensor,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        pt1: torch.Tensor,
        pt2: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_embeddings = self.layer(image_embeddings)
        src_embeddings = self.layer(src_embeddings)
        # x = self.layer(x)

        p1 = self.p1_tokens.weight.unsqueeze(0).expand(pt1.size(0), -1, -1)
        p2 = self.p2_tokens.weight.unsqueeze(0).expand(pt1.size(0), -1, -1)

        p1_tokens = torch.cat((p1, pt1), dim=1)
        p2_tokens = torch.cat((p2, pt2), dim=1)
        # print("Template tokens shape", p1_tokens.shape)
        # print("Image tokens shape", image_embeddings.shape)

        if image_embeddings.shape[0] != p1_tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, p1_tokens.shape[0], dim=0)
        else:
            src = image_embeddings
            # print("No interpolation needed")
        src = src.permute(0, 3, 1 ,2)
        img = src_embeddings.permute(0, 3, 1 ,2)
        x = x.permute(0, 3, 1 ,2)
        src = src + dense_prompt_embeddings
        # print("pos shape", image_pe.shape)
        pos_src = torch.repeat_interleave(image_pe, p1_tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        return x, img, src, pos_src, p1_tokens, p2_tokens
    
class OnePromptFormer(nn.Module):
    def __init__(self,
                 *, 
                 depth: int = 4,
                 prompt_embed_dim: int = 256,
                 embed_dim: int = 768,
                 out_chans: int = 256,
                 token_num: int,
                 patch_size: int,
                 mlp_dim: int = 1024) -> None:
        super().__init__()
        self.depth = depth
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )
        self.of = nn.ModuleList()
        self.deals = nn.ModuleList()
        
        for i in range(depth):
            self.of.append(
                _OnePromptFormer(
                    embedding_dim = embed_dim, 
                    prompt_embed_dim = prompt_embed_dim,
                    token_num = token_num, 
                    num_heads = 2, 
                    mlp_dim = mlp_dim
                                )
            )
            
            self.deals.append(
                Decode_Align(embed_dim=embed_dim, 
                             transformer_dim=prompt_embed_dim, 
                             stages=token_num-1) 
            )
            
            self.patch_embed = PatchEmbed(
                                    kernel_size=(patch_size, patch_size),
                                    stride=(patch_size, patch_size),
                                    in_chans=prompt_embed_dim,
                                    embed_dim=out_chans,
            )
            
    def forward(
        self,
        skips_raw: list,
        skips_tmp: list,
        raw_emb: torch.Tensor,
        tmp_emb: torch.Tensor,
        pt1: torch.Tensor,
        pt2: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = raw_emb + tmp_emb
        # print("x(raw+tmp) shape at begin: ", x.shape)
        x = self.neck(x.permute(0, 3, 1, 2))
        x = x.permute(0, 2, 3, 1)
        # print("x(raw+tmp->neck-> x)) shape at begin: ", x.shape)
        raw_emb = self.neck(raw_emb.permute(0, 3, 1, 2))
        # raw_emb = raw_emb.permute(0, 2, 3, 1)
        # print("Raw_emd at begin", raw_emb.shape)
        for u in range(self.depth):
            if u == 0:
                x, img_embed, tmp_embed, temp_pos,  p1, p2= self.deals[u](x, skips_raw[-(u + 1)], skips_tmp[-(u + 1)], image_pe, pt1, pt2, dense_prompt_embeddings)
                # print("x shape after 1st deal", x.shape)
                # print('tmp_embed size', tmp_embed.size())
                # print('temp_pos size', temp_pos.size())
                # print('p1 size', p1.size())
                # print('p2 size', p2.size())
                p1 = p1 + temp_pos.flatten(2).permute(0, 2, 1)
                p2 = p2 + temp_pos.flatten(2).permute(0, 2, 1)
                img_embed = img_embed.flatten(2).permute(0, 2, 1)
                tmp_embed = tmp_embed.flatten(2).permute(0, 2, 1)
                x = x.flatten(2).permute(0, 2, 1)
            # print('tmp_embed size', tmp_embed.size())
            # print('temp_pos size', temp_pos.size())
            # print('p1 size', p1.size())
            # print('p2 size', p2.size())
            x = self.of[u](x,img_embed, tmp_embed, p1, p2)
            gc.collect()
            torch.cuda.empty_cache()
            # print(x.size())
        x = rearrange(x,'b (c1 c2) d -> b d c1 c2', c1 = int(math.sqrt(x.size(1))))
        x = self.patch_embed(x)
        x = rearrange(x,'b c1 c2 d-> b (c1 c2) d')
        # print("x shape after 1 former", x.shape)
        return raw_emb, x
