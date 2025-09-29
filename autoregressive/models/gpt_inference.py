# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.drop_path import DropPath


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    factorized_n_layer: int = 5
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
            self, x: torch.Tensor, freqs_cis: torch.Tensor = None,
            input_pos: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            return_attn: bool = False
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)


        attn_scores = torch.matmul(xq, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        if self.training and self.attn_dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout_p)
        output = torch.matmul(attn_weights, values)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        output = self.resid_dropout(self.wo(output))
        if return_attn:
            return output, attn_weights
        else:
            return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None, return_attn: bool = False):
        if return_attn:
            attn_out, attn_weights = self.attention(self.attention_norm(x), freqs_cis, start_pos, mask, return_attn=True)
            h = x + self.drop_path(attn_out)
            out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
            return out, attn_weights
        else:
            h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
            out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
            return out


class WindowExtractor:
    def __init__(self, H=24, W=24):
        self.H = H
        self.W = W
        self.total_pos = H * W

        self._init_static_indices()

        self.h_cache = None
        self.current_step = 0

    def _init_static_indices(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.static_indices = torch.zeros(self.total_pos, 13, dtype=torch.long, device=device)
        self.static_mask = torch.zeros(self.total_pos, 13, dtype=torch.bool, device=device)

        for i in range(self.H):
            for j in range(self.W):
                pos = i * self.W + j
                idx = 0

                if i - 2 >= 0:
                    for dj in range(-2, 3):
                        nj = j + dj
                        if 0 <= nj < self.W:
                            self.static_indices[pos, idx] = (i - 2) * self.W + nj
                            self.static_mask[pos, idx] = True
                        idx += 1
                else:
                    idx += 5

                if i - 1 >= 0:
                    for dj in range(-2, 3):
                        nj = j + dj
                        if 0 <= nj < self.W:
                            self.static_indices[pos, idx] = (i - 1) * self.W + nj
                            self.static_mask[pos, idx] = True
                        idx += 1
                else:
                    idx += 5

                for dj in [2, 1]:
                    nj = j - dj
                    if nj >= 0:
                        self.static_indices[pos, idx] = i * self.W + nj
                        self.static_mask[pos, idx] = True
                    idx += 1

                self.static_indices[pos, 12] = pos
                self.static_mask[pos, 12] = True

    def reset(self):
        self.h_cache = None
        self.current_step = 0

    def _update_cache(self, new_h):
        if self.h_cache is None:
            self.h_cache = new_h
        else:
            self.h_cache = torch.cat([self.h_cache, new_h], dim=1)
        self.current_step += 1

    def extract(self, h, mode='train'):
        if mode == 'train':
            windows = torch.gather(
                h.unsqueeze(2).expand(-1, -1, 13, -1),
                dim=1,
                index=self.static_indices[None, ..., None].expand(h.size(0), -1, -1, h.size(-1))
            )
            return windows * self.static_mask[None, ..., None].float()

        elif mode == 'inference':
            assert h.size(1) == 1

            self._update_cache(h)
            current_pos = self.current_step - 1

            batch, _, C = h.shape
            windows = torch.zeros(batch, 1, 13, C, dtype=torch.bfloat16, device=h.device)

            indices = self.static_indices[current_pos]
            valid_mask = (indices <= current_pos) & self.static_mask[current_pos]
            valid_indices = indices[valid_mask]

            if valid_indices.numel() > 0:
                windows[:, 0, valid_mask] = torch.gather(
                    self.h_cache,
                    dim=1,
                    index=valid_indices.view(1, -1, 1).expand(batch, -1, C)
                )

            return windows


    def __call__(self, h, mode='train'):
        return self.extract(h, mode=mode)


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")

        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        factorized_dpr = [x.item() for x in
                          torch.linspace(0, self.config.drop_path_rate, self.config.factorized_n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(self.config, dpr[layer_id]))
        self.factorized_blocks = nn.ModuleList()
        for idx in range(self.config.factorized_n_layer):
            self.factorized_blocks.append(TransformerBlock(self.config, factorized_dpr[idx]))
        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.sub_vocab_size = 4096
        self.cluster_num = 256
        self.tok_embeddings1 = nn.Embedding(self.cluster_num, config.dim)
        self.tok_embeddings2 = nn.Embedding(self.sub_vocab_size, config.dim)

        self.aggregation_layer = nn.Sequential(
            nn.Linear(config.dim * 2, config.dim * 4),
            nn.GELU(),
            nn.Linear(config.dim * 4, config.dim)
        )

        self.output1 = nn.Sequential(
            nn.Linear(config.dim, config.dim * 2),
            nn.GELU(),
            nn.Linear(config.dim * 2, self.cluster_num, bias=False)
        )
        self.output2 = nn.Sequential(
            nn.Linear(config.dim, config.dim * 2),
            nn.GELU(),
            nn.Linear(config.dim * 2, self.sub_vocab_size, bias=False)
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(config.dim, config.dim // 4),
            nn.GELU(),
            nn.Linear(config.dim // 4, config.dim // 12, bias=False)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(config.dim // 12 * 12, config.dim * 2),
            nn.GELU(),
            nn.Linear(config.dim * 2, config.dim, bias=False)
        )

        self.extractor = WindowExtractor()

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head,
                                                 self.config.rope_base, self.cls_token_num)

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()
        self.attn_record = []

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output1[-1].weight, 0)
        nn.init.constant_(self.output2[-1].weight, 0)


    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)
        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head,
                                                 self.config.rope_base, self.cls_token_num)

        self.attn_record = []

    def setup_factorized_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = max_seq_length + 1

        self.max_batch_size = max_batch_size
        max_seq_length = find_multiple(max_seq_length, 8)
        for b in self.factorized_blocks:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

    def clear_caches(self):
        for b in self.layers:
            b.attention.kv_cache = None

    def forward(
            self,
            idx: torch.Tensor,
            cond_idx: torch.Tensor,
            input_pos: Optional[torch.Tensor] = None,
            targets: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            use_tok_embeddings1=None,
    ):
        # print(idx.shape,targets.shape,cond_idx.shape)

        if idx is not None and cond_idx is not None:
            idx1, idx2 = idx[0], idx[1]
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:, :self.cls_token_num]
            token_embeddings1 = self.tok_embeddings1(idx1)
            token_embeddings2 = self.tok_embeddings2(idx2)
            token_embeddings = torch.cat([token_embeddings1, token_embeddings2], dim=-1)
            token_embeddings = self.aggregation_layer(token_embeddings)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)[:, :-1, :]

            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
        else:
            if cond_idx is not None:
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:, :self.cls_token_num]
            else:
                if use_tok_embeddings1 is not None:
                    token_embeddings = self.tok_embeddings1(idx)
                else:
                    token_embeddings = self.tok_embeddings2(idx)

            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis

        B, N, D = h.shape

        if self.training:
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]

        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)

        window = self.extractor.extract(h)
        first_12 = window[:, :, :12, :]


        first_12_permuted = first_12.permute(2, 0, 1, 3)
        first_12_reshaped = first_12_permuted.reshape(12, B * N, D)

        compressed = self.mlp1(first_12_reshaped)

        compressed = compressed.reshape(12, B, N, -1)
        compressed = compressed.permute(1, 2, 0, 3)
        compressed = compressed.reshape(B, N, -1)
        compressed = self.mlp2(compressed)

        last_1 = window[:, :, -1:, :]

        compressed_expanded = compressed.unsqueeze(2)
        output = torch.cat([compressed_expanded, last_1], dim=2)
        expanded_token_embeddings = token_embeddings1.unsqueeze(2)
        factorized_ctx = torch.cat([output, expanded_token_embeddings], dim=2)
        if not factorized_ctx.is_contiguous():
            factorized_ctx = factorized_ctx.contiguous()
        factorized_ctx = factorized_ctx.view(B * N, -1, D)
        factorized_ctx_freqs_cis = freqs_cis[:factorized_ctx.shape[1]]
        for block in self.factorized_blocks:
            factorized_ctx = block(factorized_ctx, factorized_ctx_freqs_cis, mask)
        if not factorized_ctx.is_contiguous():
            factorized_ctx = factorized_ctx.contiguous()
        h = factorized_ctx.view(B, N, -1, D)

        h = self.norm(h)  # B*T*C
        logits_1 = self.output1(h[:, :, -2, :]).float()
        logits_2 = self.output2(h[:, :, -1, :]).float()

        if self.training:
            logits_1 = logits_1[:, self.cls_token_num - 1:].contiguous()
            logits_2 = logits_2[:, self.cls_token_num - 1:].contiguous()

        loss_1 = None
        loss_2 = None
        if targets is not None:
            targets1, targets2 = targets[0], targets[1]
            loss_1 = F.cross_entropy(logits_1.view(-1, logits_1.size(-1)), targets1.contiguous().view(-1))
            loss_2 = F.cross_entropy(logits_2.view(-1, logits_2.size(-1)), targets2.contiguous().view(-1))

        return (logits_1, logits_2), (loss_1, loss_2)

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)

    def generate_context(self, idx, input_pos=None, first_step=False):
        assert not self.training
        if first_step:
            token_embeddings = self.cls_embedding(idx, train=self.training)
            token_embeddings = token_embeddings.squeeze(-2)
        else:  ## the next token input
            idx_pre, idx_post = idx[0], idx[1]
            token_embeddings_vis = self.tok_embeddings1(idx_pre)
            token_embeddings_sem = self.tok_embeddings2(idx_post)
            token_embeddings = torch.cat((token_embeddings_vis, token_embeddings_sem), dim=-1)
            token_embeddings = self.aggregation_layer(token_embeddings)
        bs, N, D = token_embeddings.shape

        mask = self.causal_mask[:bs, None, input_pos]
        h = self.tok_dropout(token_embeddings)
        freq_cis = self.freqs_cis[input_pos]
        freq_cis = freq_cis.to(h.device)
        attn_weights = None
        for i, block in enumerate(self.layers):
            if i == len(self.layers) - 1:
                h, attn_weights = block(h, freq_cis, input_pos, mask, return_attn=True)
            else:
                h = block(h, freq_cis, input_pos, mask)

        if attn_weights is not None:
            attn= attn_weights[:bs//2, :, 0, 0].mean(dim=1)
            self.attn_record.append(attn)
        return h

    def decode_subtoken(self, h, x, input_pos=None, first_step=False):
        B, N, D = h.shape
        if not h.is_contiguous():
            h = h.contiguous()
        if first_step:
            window = self.extractor(h, mode='inference')
            first_12 = window[:, :, :12, :]

            first_12_permuted = first_12.permute(2, 0, 1, 3)
            first_12_reshaped = first_12_permuted.reshape(12, B * N, D)

            compressed = self.mlp1(first_12_reshaped)

            compressed = compressed.reshape(12, B, N, -1)
            compressed = compressed.permute(1, 2, 0, 3)
            compressed = compressed.reshape(B, N, -1)
            compressed = self.mlp2(compressed)
            #

            compressed_expanded = compressed.unsqueeze(2)
            factorized_ctx = compressed_expanded.view(B * N, -1, D)
            mask = self.causal_mask[:B, None, input_pos]
            factorized_ctx_freqs_cis = self.freqs_cis[input_pos]
            factorized_ctx_freqs_cis = factorized_ctx_freqs_cis.to(h.device)

            for block in self.factorized_blocks:
                factorized_ctx = block(factorized_ctx, factorized_ctx_freqs_cis, start_pos=input_pos, mask=mask)

            factorized_ctx = h.reshape(B * N, -1, D)

        else:
            idx = x[0]
            token_embedding = self.tok_embeddings1(idx)
            factorized_ctx = token_embedding.reshape(B * N, -1, D)

        mask = self.causal_mask[:B, None, input_pos + 1]
        factorized_ctx_freqs_cis = self.freqs_cis[input_pos + 1]
        factorized_ctx_freqs_cis = factorized_ctx_freqs_cis.to(h.device)

        for block in self.factorized_blocks:
            factorized_ctx = block(factorized_ctx, factorized_ctx_freqs_cis, start_pos=input_pos + 1, mask=mask)

        h = factorized_ctx.reshape(B, N, -1, D)
        h = self.norm(h)

        logits = self.output1(h[:, :, 0, :]) if first_step else self.output2(h[:, :, 0, :])

        return logits


#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)  # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache])  # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)],
                             dim=-1)  # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cache = cache.repeat_interleave(2, dim=0)  # repeat 双份
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache])  # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)  # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
        xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
        xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs))


def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs))


def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs))


def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs))


def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs))


def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs))


def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs))


def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs))


GPT_models = {
    'GPT-B': GPT_B, 'ht-GPT-L': GPT_L, 'ht-GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'ht-GPT-3B': GPT_3B, 'ht-GPT-7B': GPT_7B,
}