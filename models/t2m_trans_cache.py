import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding

class Text2Motion_Transformer(nn.Module):
    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                semantic_interleaved_flag=False,
                dual_head_flag=False,
                semantic_len=50,
                uncond_prob=0):
        super().__init__()
        
        if dual_head_flag:
            self.trans_base = CrossCondTransDualBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, semantic_len, uncond_prob)
            self.trans_head = CrossCondTransDualHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, semantic_len)
        else:
            raise NotImplementedError("Only dual_head_flag=True is implemented in this version.")
        self.block_size = block_size
        self.num_vq = num_vq
        self.semantic_interleaved_flag = semantic_interleaved_flag
        self.dual_head_flag = dual_head_flag
        self.semantic_len = semantic_len
        self.uncond_prob = uncond_prob
        self.num_layers = num_layers
        self.n_head = n_head
        self.head_size = embed_dim // n_head
        # 总层数是base和head的层数之和
        self.total_layers = num_layers * 2
        _assumed_codebook_part_size = (self.num_vq - 1) // 2 # e.g. (1025-1)//2 = 512

        self.SEMANTIC_TOKEN_END_IDX = _assumed_codebook_part_size - 1
        self.SEPARATOR_TOKEN_IDX = _assumed_codebook_part_size
        self.RECONSTRUCTION_TOKEN_START_IDX = _assumed_codebook_part_size + 1

    def get_block_size(self):
        return self.block_size

    def forward(self, idxs, clip_feature, semantic_valid_lengths=None, cfg_mode=False, kv_cache=None):
        B, T_sequence = idxs.shape
        device = idxs.device

        key_padding_mask_for_idxs = torch.zeros_like(idxs, dtype=torch.bool) # Default to False (not masked)

        if semantic_valid_lengths is not None:
            for i in range(B):
                valid_sem_len = semantic_valid_lengths[i].item()
                valid_sem_len_with_end = valid_sem_len + 1
                actual_semantic_segment_end = self.semantic_len
                
                if valid_sem_len_with_end < actual_semantic_segment_end:
                    key_padding_mask_for_idxs[i, valid_sem_len_with_end:actual_semantic_segment_end] = True
        
        final_attention_mask_for_trans_head_input = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device), # False for condition embedding
            key_padding_mask_for_idxs
        ], dim=1)

        if semantic_valid_lengths is not None:
            feat = self.trans_base(idxs, clip_feature, key_padding_mask=final_attention_mask_for_trans_head_input, cfg_mode=cfg_mode, kv_cache=kv_cache)
            logits = self.trans_head(feat, key_padding_mask=final_attention_mask_for_trans_head_input, kv_cache=kv_cache)
        else:
            feat = self.trans_base(idxs, clip_feature, cfg_mode=cfg_mode, kv_cache=kv_cache)
            logits = self.trans_head(feat, kv_cache=kv_cache)
        return logits

    def sample_batch(self, clip_feature, if_categorial=False, cond_scale=1.0, min_motion_len=0, topp=0.96, temperature_sem=None, temperature_motion=None):
        device = clip_feature.device
        use_cfg = (cond_scale != 1.0)
        actual_B = clip_feature.shape[0] 
        
        if use_cfg:
            if hasattr(self.trans_base, 'uncond_embedding'):
                uncond_feature = torch.zeros_like(clip_feature)
            else:
                uncond_feature = torch.zeros_like(clip_feature)
            
            combined_clip_feature = torch.cat([clip_feature, uncond_feature], dim=0)
            full_B = actual_B * 2
        else:
            combined_clip_feature = clip_feature
            full_B = actual_B
        
        xs = torch.empty((full_B, 0), dtype=torch.long, device=device)
        all_finished = torch.zeros(full_B, dtype=torch.bool, device=device)
        
        kv_cache = KV_Cache(
            num_layers=self.num_layers, 
            batch_size=full_B,
            max_seq_len=self.block_size,
            n_head=self.n_head,
            head_size=self.head_size,
            device=device
        )
        
        if self.dual_head_flag:
            sem_ended = torch.zeros(full_B, dtype=torch.bool, device=device)
            sem_lengths = torch.full((full_B,), self.semantic_len, dtype=torch.long, device=device)
            in_sem_phase = torch.ones(full_B, dtype=torch.bool, device=device)

        for _ in range(self.block_size):
            check_indices = slice(actual_B) if use_cfg else slice(full_B)
            if all_finished[check_indices].all():
                break
            
            current_len = xs.shape[1]
            
            fwd_sem_lengths = None
            if self.dual_head_flag:
                fwd_sem_lengths = sem_lengths.clone()
                if current_len < self.semantic_len:
                    for i in range(full_B):
                        if not sem_ended[i]:
                            fwd_sem_lengths[i] = current_len
            
            logits = self.forward(xs, combined_clip_feature, semantic_valid_lengths=fwd_sem_lengths, kv_cache=kv_cache)
            
            if use_cfg:
                cond_logits = logits[:actual_B, -1, :]
                uncond_logits = logits[actual_B:, -1, :]
                x_logits = uncond_logits + cond_scale * (cond_logits - uncond_logits)
            else:
                x_logits = logits[:, -1, :]
            
            if self.dual_head_flag:
                if temperature_sem is not None and current_len < self.semantic_len:
                    x_logits = x_logits / temperature_sem
                elif temperature_motion is not None and current_len >= self.semantic_len:
                    x_logits = x_logits / temperature_motion
            else:
                if temperature_motion is not None:
                    x_logits = x_logits / temperature_motion
            
            if min_motion_len > 0:
                process_range = actual_B if use_cfg else full_B
                if self.dual_head_flag:
                    for i in range(process_range):
                        if not in_sem_phase[i]:
                            motion_len = current_len - self.semantic_len
                            if motion_len < min_motion_len:
                                x_logits[i, self.num_vq] = -float("Inf")
                else:
                    for i in range(process_range):
                        if current_len < min_motion_len:
                            x_logits[i, self.num_vq] = -float("Inf")
            
            if topp is not None:
                x_logits = top_k_top_p_filtering(x_logits, top_k=0, top_p=topp)
            probs = F.softmax(x_logits, dim=-1)
        
            if if_categorial:
                candidate_tokens = Categorical(probs).sample()
            else:
                candidate_tokens = torch.topk(probs, k=1, dim=-1)[1].squeeze(-1)
            step_tokens = torch.full((full_B,), self.num_vq + 1, dtype=torch.long, device=device)
            
            process_range = actual_B if use_cfg else full_B
            for i in range(process_range):
                if all_finished[i]:
                    continue
                
                token = candidate_tokens[i]
                
                if self.dual_head_flag:
                    step_tokens[i] = self._process_dual_head_token(
                        i, token, current_len, in_sem_phase, sem_ended, 
                        sem_lengths, all_finished
                    )
                else:
                    if token == self.num_vq:
                        step_tokens[i] = self.num_vq
                        all_finished[i] = True
                    else:
                        step_tokens[i] = token

            if use_cfg:
                step_tokens[actual_B:] = step_tokens[:actual_B]
            
            xs = torch.cat((xs, step_tokens.unsqueeze(-1)), dim=1)
            
            kv_cache.increment_len()
            
        return xs[:actual_B] if use_cfg else xs
    
    def _process_dual_head_token(self, i, token, current_len, in_sem_phase, 
                                  sem_ended, sem_lengths, all_finished):
        if in_sem_phase[i]:
            if sem_ended[i]:
                if (current_len + 1) >= self.semantic_len:
                    in_sem_phase[i] = False
                return self.num_vq + 1 
            
            if token == self.num_vq:
                sem_ended[i] = True
                sem_lengths[i] = current_len  
                if (current_len + 1) >= self.semantic_len:
                    in_sem_phase[i] = False
                return self.num_vq  
            
            if (current_len + 1) >= self.semantic_len:
                in_sem_phase[i] = False
                if not sem_ended[i]:
                    sem_ended[i] = True
            
            return token
        else:
            if token == self.num_vq:
                all_finished[i] = True
                return self.num_vq
            return token

def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits



def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    logits = logits[:, -1, :] / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    
    probs = F.softmax(logits, dim=-1)

    if not sample_logits:
        _, x = top_k(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)

    return x

class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, semantic_max_length=50):
        super().__init__()
        assert embed_dim % n_head == 0, "embed_dim must be divisible by n_head"
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head
        self.head_size = embed_dim // n_head
        self.semantic_max_length = semantic_max_length

    def forward(self, x, key_padding_mask=None, kv_cache=None, layer_idx=None):
        B, T_x, C = x.size()

        q_cur = self.query(x).view(B, T_x, self.n_head, self.head_size).transpose(1, 2)
        k_cur = self.key(x).view(B, T_x, self.n_head, self.head_size).transpose(1, 2)
        v_cur = self.value(x).view(B, T_x, self.n_head, self.head_size).transpose(1, 2)
        
        if kv_cache is not None and layer_idx is not None:
            k_cache, v_cache = kv_cache.get_kv(layer_idx)

            kv_cache.update(layer_idx, k_cur, v_cur)

            k = torch.cat([k_cache, k_cur], dim=2)
            v = torch.cat([v_cache, v_cur], dim=2)

            q = q_cur[:, :, -1:, :]
        else:
            k = k_cur
            v = v_cur
            q = q_cur

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Causal mask logic based on T_x and k's total length (T_k = k.size(2))
        T_k = k.size(2)

        query_indices_in_full_sequence = torch.arange(T_k - T_x, T_k, device=x.device).view(T_x, 1)
        key_indices = torch.arange(T_k, device=x.device).view(1, T_k)
        causal_condition_mask = key_indices <= query_indices_in_full_sequence # Shape (T_x, T_k)
        
        att = att.masked_fill(~causal_condition_mask.unsqueeze(0).unsqueeze(0), float('-inf')) # Expand to (1,1,T_x,T_k)

        if key_padding_mask is not None:
            full_key_padding_mask = key_padding_mask 
            
            assert full_key_padding_mask.shape[1] == T_k, f"full_key_padding_mask shape {full_key_padding_mask.shape} last dim must match K's length {T_k}"
            expanded_key_padding_mask = full_key_padding_mask.unsqueeze(1).unsqueeze(2) 
            att = att.masked_fill(expanded_key_padding_mask, float('-inf'))
            
        att = F.softmax(att, dim=-1)
        
        att = self.attn_drop(att)
        y = att @ v 

        y = y.transpose(1, 2).contiguous().view(B, T_x, C)
        y = self.resid_drop(self.proj(y))

        return y

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, key_padding_mask=None, kv_cache=None, layer_idx=None):
        # Attention part
        attn_out = self.attn(self.ln1(x), key_padding_mask=key_padding_mask, 
                                kv_cache=kv_cache, layer_idx=layer_idx)
        
        x = x + attn_out # Residual connection
        
        # MLP part
        x = x + self.mlp(self.ln2(x))
        
        return x

class CrossCondTransDualBase(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                semantic_len = 50,
                uncond_prob=0):
        super().__init__()
        self.tok_emb = nn.ModuleList([nn.Embedding(num_vq + 2, embed_dim) for _ in range(2)])
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        if uncond_prob > 0:
            self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(1, embed_dim) / embed_dim ** 0.5))
        self.semantic_len = semantic_len
        self.uncond_prob = uncond_prob
        self.embed_dim = embed_dim
        # transformer block
        self.blocks = nn.ModuleList([Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size
        self.num_layers = num_layers
        self.n_head = n_head
        self.head_size = embed_dim // n_head

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, clip_feature, key_padding_mask=None, cfg_mode=False, kv_cache=None):
        key_padding_mask_for_idxs = key_padding_mask 

        condition_embedding = self.cond_emb(clip_feature).unsqueeze(1) # (B, 1, C)
        mask_for_blocks = None
        if self.uncond_prob > 0 and self.training:
            uncond_prob = torch.rand(clip_feature.shape[0], device=clip_feature.device)
            uncond_prob = uncond_prob < self.uncond_prob
            uncond_prob = uncond_prob.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
            condition_embedding = torch.where(uncond_prob, self.uncond_embedding, condition_embedding)

        if cfg_mode:
            uncond_feat = self.uncond_embedding.repeat(clip_feature.shape[0] // 2, 1).unsqueeze(1)
            condition_embedding = torch.cat([condition_embedding[:clip_feature.shape[0] // 2], uncond_feat], dim=0)

        if not self.training and kv_cache is not None and idx.shape[1] > 0:
            if len(idx) > 0:
                B, T_idx = idx.size()
                last_token = idx[:, -1:]
                
                if T_idx <= self.semantic_len:
                    token_embeddings = self.tok_emb[0](last_token)
                else:
                    token_embeddings = self.tok_emb[1](last_token)
                    
                position_ids = torch.full((B, T_idx + 1, self.embed_dim), 0, device=idx.device)
                position_embeddings = self.pos_embed(position_ids)
                
                x = token_embeddings + position_embeddings[:, -1:, :]
                    
                for i, block in enumerate(self.blocks):
                    x = block(x, key_padding_mask=key_padding_mask, kv_cache=kv_cache, layer_idx=i)
                    
                return x    
        
        if len(idx) == 0:
            token_embeddings = condition_embedding
            # Mask for blocks is just for the condition embedding (i.e., False, do not mask)
            mask_for_blocks = torch.zeros(condition_embedding.shape[0], 1, dtype=torch.bool, device=condition_embedding.device)
        else:
            b, t_idx = idx.size()
            assert t_idx <= self.block_size, f"Cannot forward, idx sequence length {t_idx} exceeds model block size {self.block_size}."

            # Prepare token embeddings from idx based on whether it's semantic, reconstruction, or both
            if t_idx <= self.semantic_len : # Only semantic tokens (or shorter than self.semantic_len)
                token_embeddings_unconditioned = self.tok_emb[0](idx)
            else: # Both semantic and reconstruction tokens
                token_sem_embeddings = self.tok_emb[0](idx[..., :self.semantic_len])
                token_recon_embeddings = self.tok_emb[1](idx[..., self.semantic_len:t_idx]) # Slice up to t_idx
                token_embeddings_unconditioned = torch.cat([token_sem_embeddings, token_recon_embeddings], dim=1)
            
            token_embeddings = torch.cat([condition_embedding, token_embeddings_unconditioned], dim=1)
            
            # Adjust key_padding_mask_for_idxs for the prepended condition embedding
            if key_padding_mask_for_idxs is not None:
                # Ensure key_padding_mask_for_idxs matches the length of idx used for embeddings
                key_padding_mask_for_idxs = key_padding_mask_for_idxs[:, :t_idx + 1]
            else:
                # If no original mask, then no padding for any part of the sequence passed to blocks (beyond causal)
                # Create a mask of all Falses for the blocks if none provided.
                mask_for_blocks = torch.zeros(token_embeddings.shape[0], token_embeddings.shape[1], dtype=torch.bool, device=idx.device)
        
        
        
        x = self.pos_embed(token_embeddings)
        for i, block in enumerate(self.blocks):
            x = block(x, key_padding_mask=mask_for_blocks, kv_cache=kv_cache, layer_idx=i)
            
        
        return x

    
class CrossCondTransDualHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                semantic_len = 50):
        super().__init__()

        self.blocks = nn.ModuleList([Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])
        self.sem_heads = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.recon_heads = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.semantic_len = semantic_len
        self.num_vq = num_vq
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, key_padding_mask=None, kv_cache=None):
        key_padding_mask_for_input_x = key_padding_mask

        kv_cache_offset = 0
        if kv_cache is not None:
            kv_cache_offset = len(self.blocks)

        for i, block in enumerate(self.blocks):
            layer_idx = i + kv_cache_offset if kv_cache is not None else None
            x = block(x, key_padding_mask=key_padding_mask_for_input_x, kv_cache=kv_cache, layer_idx=layer_idx)
        if kv_cache is not None:
            current_len = kv_cache.current_len + 1
            if current_len <= self.semantic_len:
                x_semantic_out = self.ln_f[0](x)
                logits_semantic = self.sem_heads(x_semantic_out)
                logits_result = logits_semantic
                return logits_result
            else:
                x_recon_out = self.ln_f[1](x)
                logits_recon = self.recon_heads(x_recon_out)
                logits_result = logits_recon
                return logits_result
        else:
            x_semantic_part = x[:, :self.semantic_len, :]
            x_recon_part = x[:, self.semantic_len:, :] # Takes the rest
            
            logits_semantic = torch.empty(x.shape[0], 0, self.sem_heads.out_features, device=x.device)
            if x_semantic_part.shape[1] > 0:
                x_semantic_out = self.ln_f[0](x_semantic_part)
                logits_semantic = self.sem_heads(x_semantic_out)
            
            logits_recon = torch.empty(x.shape[0], 0, self.recon_heads.out_features, device=x.device)
            if x_recon_part.shape[1] > 0:
                x_recon_out = self.ln_f[1](x_recon_part)
                logits_recon = self.recon_heads(x_recon_out)

            logits_result = torch.cat([logits_semantic, logits_recon], dim=1)
            return logits_result

class KV_Cache:
    def __init__(self, num_layers, batch_size, max_seq_len, n_head, head_size, device):
        self.num_layers = num_layers * 2
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_head = n_head
        self.head_size = head_size
        self.device = device

        self.reset()
    
    def reset(self):
        self.k_cache = [torch.zeros(self.batch_size, self.n_head, self.max_seq_len, self.head_size, device=self.device) 
                       for _ in range(self.num_layers)]
        self.v_cache = [torch.zeros(self.batch_size, self.n_head, self.max_seq_len, self.head_size, device=self.device) 
                       for _ in range(self.num_layers)]
        self.current_len = 0
    
    def update(self, layer_idx, k, v):
        self.k_cache[layer_idx][:, :, self.current_len:self.current_len + k.size(2), :] = k
        self.v_cache[layer_idx][:, :, self.current_len:self.current_len + v.size(2), :] = v
    
    def get_kv(self, layer_idx):
        return self.k_cache[layer_idx][:, :, :self.current_len, :], self.v_cache[layer_idx][:, :, :self.current_len, :]
    
    def increment_len(self, increment=1):
        self.current_len += increment

        

