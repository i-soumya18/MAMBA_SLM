"""
Extended Context Support for Hybrid Mamba-Transformer
Supports up to 4K tokens with optimized positional encoding
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    More efficient for longer contexts than absolute positional encoding
    """
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Build cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len: int):
        """Build rotation matrix cache"""
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        """
        Apply rotary positional encoding
        
        Args:
            x: Input tensor [batch, n_heads, seq_len, head_dim]
            seq_len: Sequence length (uses x.shape[2] if None)
            
        Returns:
            Tensor with rotary encoding applied
        """
        if seq_len is None:
            seq_len = x.shape[2]
        
        # Extend cache if needed
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        
        return (
            x * self.cos_cached[:, :, :seq_len, :x.shape[-1]] +
            self._rotate_half(x) * self.sin_cached[:, :, :seq_len, :x.shape[-1]]
        )
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)


class ALiBiPositionalEncoding(nn.Module):
    """
    Attention with Linear Biases (ALiBi)
    Alternative to RoPE, adds position-dependent bias to attention scores
    """
    
    def __init__(self, n_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        # Compute slopes for each head
        slopes = self._get_slopes(n_heads)
        self.register_buffer('slopes', slopes)
        
        # Build position bias matrix
        self._build_cache(max_seq_len)
    
    def _get_slopes(self, n_heads: int) -> torch.Tensor:
        """Get geometric sequence of slopes for each attention head"""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return torch.tensor([start * (ratio ** i) for i in range(n)])
        
        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = self._get_slopes(2 * closest_power_of_2)[::(2 * closest_power_of_2 // n_heads)]
            slopes = torch.cat([slopes_a, slopes_b[:n_heads - closest_power_of_2]])
            return slopes
    
    def _build_cache(self, max_seq_len: int):
        """Build position bias matrix"""
        # Create position indices
        context_position = torch.arange(max_seq_len)[:, None]
        memory_position = torch.arange(max_seq_len)[None, :]
        relative_position = memory_position - context_position
        
        # Apply slopes
        biases = relative_position[None, :, :] * self.slopes[:, None, None]
        self.register_buffer('bias_cached', biases, persistent=False)
    
    def forward(self, attention_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Add position bias to attention scores
        
        Args:
            attention_scores: Attention scores [batch, n_heads, seq_len, seq_len]
            seq_len: Current sequence length
            
        Returns:
            Biased attention scores
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        return attention_scores + self.bias_cached[:, :seq_len, :seq_len]


class ExtendedContextAttention(nn.Module):
    """
    Attention block optimized for extended context (up to 4K tokens)
    Uses RoPE or ALiBi for better position encoding
    """
    
    def __init__(self,
                 d_model: int,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 4096,
                 position_encoding: str = 'rope'):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            position_encoding: Type of position encoding ('rope', 'alibi', or 'absolute')
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.position_encoding = position_encoding
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Position encoding
        if position_encoding == 'rope':
            self.pos_encoder = RotaryPositionalEncoding(self.head_dim, max_seq_len)
        elif position_encoding == 'alibi':
            self.pos_encoder = ALiBiPositionalEncoding(n_heads, max_seq_len)
        else:
            self.pos_encoder = None
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, 
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with extended context support
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = x.shape
        
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq_len, head_dim]
        
        # Apply positional encoding
        if self.position_encoding == 'rope':
            q = self.pos_encoder(q, seq_len)
            k = self.pos_encoder(k, seq_len)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply ALiBi if using
        if self.position_encoding == 'alibi':
            scores = self.pos_encoder(scores, seq_len)
        
        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, -1e9)
        
        # Apply custom attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Attention weights and output
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        
        x = residual + self.dropout(attn_output)
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention for very long sequences
    Reduces memory complexity from O(n²) to O(n*w) where w is window size
    """
    
    def __init__(self,
                 d_model: int,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 window_size: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        
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
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sliding window attention
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = x.shape
        
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        # Sliding window attention
        attn_output = self._sliding_window_attention(q, k, v)
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        
        x = residual + self.dropout(attn_output)
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x
    
    def _sliding_window_attention(self, q, k, v):
        """Compute attention with sliding window"""
        batch_size, n_heads, seq_len, head_dim = q.shape
        output = torch.zeros_like(q)
        
        for i in range(seq_len):
            # Define window
            start = max(0, i - self.window_size + 1)
            end = i + 1
            
            # Attention for this position
            q_i = q[:, :, i:i+1, :]  # [batch, n_heads, 1, head_dim]
            k_window = k[:, :, start:end, :]  # [batch, n_heads, window, head_dim]
            v_window = v[:, :, start:end, :]
            
            scores = torch.matmul(q_i, k_window.transpose(-2, -1)) * self.scale
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output[:, :, i:i+1, :] = torch.matmul(attn_weights, v_window)
        
        return output


def upgrade_model_context_length(model: nn.Module,
                                new_max_length: int = 4096,
                                position_encoding: str = 'rope'):
    """
    Upgrade model to support extended context length
    
    Args:
        model: Original model
        new_max_length: New maximum sequence length
        position_encoding: Position encoding type
        
    Returns:
        Model with extended context support
    """
    # Update max_seq_length attribute
    model.max_seq_length = new_max_length
    
    # Update position embeddings if using absolute encoding
    if hasattr(model, 'embed_positions'):
        old_pos_emb = model.embed_positions
        new_pos_emb = nn.Embedding(new_max_length, model.d_model)
        
        # Copy existing embeddings
        with torch.no_grad():
            old_len = old_pos_emb.weight.shape[0]
            new_pos_emb.weight[:old_len] = old_pos_emb.weight
            
            # Interpolate for new positions
            if new_max_length > old_len:
                # Simple interpolation
                for i in range(old_len, new_max_length):
                    idx = int((i / new_max_length) * old_len)
                    new_pos_emb.weight[i] = old_pos_emb.weight[min(idx, old_len-1)]
        
        model.embed_positions = new_pos_emb
    
    print(f"Model context length extended to {new_max_length} tokens")
    print(f"Position encoding: {position_encoding}")
    
    return model


if __name__ == "__main__":
    print("Extended Context Support Module")
    print("\nPosition Encoding Options:")
    print("  - RoPE (Rotary): Best for general use, efficient")
    print("  - ALiBi: Good extrapolation to longer contexts")
    print("  - Sliding Window: Memory-efficient for very long sequences")
    print("\nContext Length: 1K → 4K tokens")
    print("Expected memory increase: ~4x for attention (with optimizations)")
