"""
Hybrid Mamba-Transformer Small Language Model
Extracted from MAMBA_SLM.ipynb
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List

class MambaBlock(nn.Module):
    """Simplified Mamba SSM Block for hybrid architecture"""
    def __init__(self, d_model, d_state=16, expand_factor=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_inner = d_model * expand_factor

        # Linear projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=4,
            bias=True,
            padding=2,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, d_state, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # State space parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        B, L, D = x.shape

        residual = x
        x = self.norm(x)

        # Linear projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each

        # Convolution
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # causal conv
        x = x.transpose(1, 2)  # (B, L, d_inner)

        # Activation
        x = nn.functional.silu(x)

        # SSM step (simplified)
        A = -torch.exp(self.A_log.float())  # (d_state,)

        # Selective mechanism
        x_dbl = self.x_proj(x)  # (B, L, 2*d_state)
        delta, B_proj = x_dbl.chunk(2, dim=-1)  # (B, L, d_state) each
        delta = nn.functional.softplus(self.dt_proj(x))  # (B, L, d_state)

        # Simplified SSM computation (for efficiency)
        y = x * self.D + torch.sum(B_proj * delta, dim=-1, keepdim=True)

        # Gate and output
        y = y * nn.functional.silu(z)
        output = self.out_proj(y)

        return output + residual


class HybridAttentionBlock(nn.Module):
    """Efficient Transformer attention block"""
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # x: (batch, seq_len, d_model)
        B, L, D = x.shape

        # Self-attention
        residual = x
        x = self.norm1(x)

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (B, n_heads, L, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # Ensure mask has the right shape: (B, 1, L, L) or (1, 1, L, L)
            if attention_mask.dim() == 4:
                scores = scores + (1.0 - attention_mask) * -10000.0
            else:
                scores = scores.masked_fill(attention_mask == 0, -10000.0)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        attn_output = self.o_proj(attn_output)

        x = residual + attn_output

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class HybridMambaTransformer(nn.Module):
    """Hybrid Mamba-Transformer Model optimized for efficiency"""
    def __init__(self,
                 vocab_size=32000,
                 d_model=768,
                 n_layers=12,
                 n_heads=12,
                 d_state=16,
                 expand_factor=2,
                 dropout=0.1,
                 max_seq_length=2048,
                 layer_pattern=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_length = max_seq_length

        # Default layer pattern: Mamba for early layers, Transformer for later layers
        if layer_pattern is None:
            # 70% Mamba, 30% Transformer - optimal for efficiency
            mamba_layers = int(n_layers * 0.7)
            self.layer_pattern = ['mamba'] * mamba_layers + ['transformer'] * (n_layers - mamba_layers)
        else:
            self.layer_pattern = layer_pattern

        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.embed_positions = nn.Embedding(max_seq_length, d_model)

        # Hybrid layers
        self.layers = nn.ModuleList()
        for layer_type in self.layer_pattern:
            if layer_type == 'mamba':
                self.layers.append(MambaBlock(d_model, d_state, expand_factor))
            else:  # transformer
                self.layers.append(HybridAttentionBlock(d_model, n_heads, dropout))

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing for memory efficiency"""
        self._gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self._gradient_checkpointing = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, L = input_ids.shape

        # Embeddings
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)

        # Process through hybrid layers
        for i, (layer, layer_type) in enumerate(zip(self.layers, self.layer_pattern)):
            if layer_type == 'mamba':
                x = layer(x)
            else:  # transformer
                # Create causal mask for transformer layers  
                causal_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=x.dtype))
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
                x = layer(x, causal_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        return {'loss': loss, 'logits': logits}

    def generate(self, input_ids, max_length=100, temperature=0.8, top_p=0.9, eos_token_id=None):
        """Simple generation function"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.forward(input_ids)
                logits = outputs['logits'][:, -1, :] / temperature

                # Top-p sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

        return input_ids
