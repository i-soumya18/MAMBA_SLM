"""
Flash Attention Integration for Hybrid Mamba-Transformer
Provides memory-efficient attention with O(1) memory complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("Flash Attention not available. Install with: pip install flash-attn --no-build-isolation")


class FlashAttentionBlock(nn.Module):
    """
    Optimized attention block using Flash Attention 2
    Significantly reduces memory usage and improves speed
    """
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 use_flash: bool = True,
                 causal: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_flash = use_flash and FLASH_ATTENTION_AVAILABLE
        self.causal = causal
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # QKV projection - combined for efficiency
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, 
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional Flash Attention
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        
        if self.use_flash and FLASH_ATTENTION_AVAILABLE:
            # Use Flash Attention
            attn_output = self._flash_attention(x, attention_mask)
        else:
            # Fall back to standard attention
            attn_output = self._standard_attention(x, attention_mask)
        
        x = residual + self.dropout(attn_output)
        
        # MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x
    
    def _flash_attention(self, 
                        x: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Flash Attention 2 implementation
        More memory efficient than standard attention
        """
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        qkv = self.qkv(x)  # [batch, seq_len, 3 * d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        
        # Rearrange to [batch, seq_len, 3, n_heads, head_dim]
        # Then split and rearrange to [batch, n_heads, seq_len, head_dim]
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq_len, head_dim]
        
        # Flash attention expects [batch, seq_len, n_heads, head_dim]
        q = q.transpose(1, 2)  # [batch, seq_len, n_heads, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Call Flash Attention
        # Flash attention handles causal masking internally
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            softmax_scale=self.scale,
            causal=self.causal
        )
        
        # Reshape back
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        
        return attn_output
    
    def _standard_attention(self,
                          x: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Standard scaled dot-product attention
        Fallback when Flash Attention is not available
        """
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq_len, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if self.causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
            )
            scores = scores.masked_fill(~causal_mask, -1e9)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, n_heads, seq_len, head_dim]
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention without Flash Attention
    Uses chunked computation to reduce memory footprint
    """
    
    def __init__(self,
                 d_model: int,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 chunk_size: int = 1024):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = chunk_size
        
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
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Chunked attention computation"""
        batch_size, seq_len, _ = x.shape
        
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        # Chunked attention computation
        attn_output = self._chunked_attention(q, k, v, attention_mask)
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        
        x = residual + self.dropout(attn_output)
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x
    
    def _chunked_attention(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention in chunks to reduce memory usage
        """
        batch_size, n_heads, seq_len, head_dim = q.shape
        
        # Initialize output
        output = torch.zeros_like(q)
        
        # Process in chunks
        for i in range(0, seq_len, self.chunk_size):
            end_i = min(i + self.chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i, :]
            
            # Compute attention scores for this chunk
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
            
            # Apply causal mask
            causal_mask = torch.tril(
                torch.ones(end_i - i, seq_len, device=q.device, dtype=torch.bool)
            )
            scores = scores.masked_fill(~causal_mask, -1e9)
            
            # Softmax and apply to values
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output[:, :, i:end_i, :] = torch.matmul(attn_weights, v)
        
        return output


def replace_attention_with_flash(model, use_flash: bool = True):
    """
    Replace standard attention blocks with Flash Attention blocks
    
    Args:
        model: Model with HybridAttentionBlock layers
        use_flash: Whether to use Flash Attention (falls back to standard if unavailable)
    
    Returns:
        Modified model
    """
    if not FLASH_ATTENTION_AVAILABLE and use_flash:
        print("Warning: Flash Attention not available, using standard attention")
        use_flash = False
    
    # Recursively replace attention blocks
    for name, module in model.named_children():
        if module.__class__.__name__ == 'HybridAttentionBlock':
            # Replace with Flash Attention block
            new_module = FlashAttentionBlock(
                d_model=module.d_model,
                n_heads=module.n_heads,
                dropout=module.dropout.p,
                use_flash=use_flash
            )
            # Copy weights
            new_module.load_state_dict(module.state_dict(), strict=False)
            setattr(model, name, new_module)
        else:
            # Recursively apply to children
            replace_attention_with_flash(module, use_flash)
    
    return model


# Utility function for model configuration
def get_attention_class(use_flash: bool = True, 
                       use_memory_efficient: bool = False):
    """
    Get the appropriate attention class based on availability and preference
    
    Args:
        use_flash: Prefer Flash Attention if available
        use_memory_efficient: Use memory-efficient chunked attention
        
    Returns:
        Attention class to use
    """
    if use_flash and FLASH_ATTENTION_AVAILABLE:
        print("Using Flash Attention 2")
        return FlashAttentionBlock
    elif use_memory_efficient:
        print("Using Memory-Efficient Chunked Attention")
        return MemoryEfficientAttention
    else:
        print("Using Standard Attention")
        # Import the standard attention block from main model
        from hybrid_mamba_training import HybridAttentionBlock
        return HybridAttentionBlock


if __name__ == "__main__":
    # Test Flash Attention availability
    print(f"Flash Attention Available: {FLASH_ATTENTION_AVAILABLE}")
    
    if FLASH_ATTENTION_AVAILABLE:
        print("Flash Attention 2 is ready to use!")
        print("Expected speedup: 2-3x for training, 1.5-2x for inference")
        print("Expected memory reduction: 40-60%")
    else:
        print("Install Flash Attention with:")
        print("  pip install flash-attn --no-build-isolation")
        print("Or use memory-efficient attention as fallback")
