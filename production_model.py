"""
Production-Grade Hybrid Mamba-Transformer Model
Implements advanced features: Flash Attention, RoPE, GQA, SwiGLU, Gradient Checkpointing
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
from production_config import ProductionModelConfig


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    More effective than absolute positional encodings for long sequences
    """
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cache(self, seq_len: int, device: torch.device):
        """Update cached cos/sin values if sequence length changed"""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for rotary embeddings"""
        self._update_cache(seq_len, x.device)
        return (
            self._cos_cached[:, :, :seq_len, :].to(x.dtype),
            self._sin_cached[:, :, :seq_len, :].to(x.dtype)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                         cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    Reduces KV cache size while maintaining quality
    Multi-Query Attention when num_key_value_heads = 1
    Multi-Head Attention when num_key_value_heads = num_heads
    """
    def __init__(self, config: ProductionModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.num_key_value_heads = config.num_key_value_heads if config.use_grouped_query_attention else config.n_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
        # Rotary embeddings
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_seq_length)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if self.config.use_rotary_embeddings:
            cos, sin = self.rotary_emb(value_states, seq_length)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat KV heads if using GQA
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Compute attention
        if self.config.use_flash_attention:
            try:
                # Try to use Flash Attention 2
                from flash_attn import flash_attn_func
                
                # Reshape for flash attention (batch, seq, heads, head_dim)
                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)
                
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout_p=self.config.attention_dropout if self.training else 0.0,
                    causal=True,
                )
                
                attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
                
            except ImportError:
                # Fallback to standard attention
                attn_output = self._standard_attention(
                    query_states, key_states, value_states, attention_mask, batch_size, seq_length
                )
        else:
            attn_output = self._standard_attention(
                query_states, key_states, value_states, attention_mask, batch_size, seq_length
            )
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value
    
    def _standard_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_length: int,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention"""
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        
        return attn_output


class SwiGLU(nn.Module):
    """
    Gated Linear Unit with Swish activation (SwiGLU)
    Used in modern LLMs like LLaMA, PaLM for better performance
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up projection
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ProductionMambaBlock(nn.Module):
    """
    Enhanced Mamba SSM Block for production
    Includes pre-normalization and residual connections
    """
    def __init__(self, config: ProductionModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = 4
        self.expand = config.expand_factor
        self.d_inner = int(self.expand * self.d_model)
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)
        
        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )
        
        # SSM projections
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_state, self.d_inner, bias=True)
        
        # SSM parameters
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, dim = hidden_states.shape
        residual = hidden_states
        
        # Pre-normalization
        if self.config.use_pre_norm:
            hidden_states = self.norm(hidden_states)
        
        # Input projection and split
        x_and_res = self.in_proj(hidden_states)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # Convolution
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # Activation
        x = F.silu(x)
        
        # SSM
        x_proj = self.x_proj(x)
        B, C = x_proj.split([self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj(torch.mean(x, dim=-1))  # (batch, seq_len, d_inner)
        dt = F.softplus(dt)
        
        # State space computation (simplified for production)
        A = -torch.exp(self.A_log.float())
        y = self._selective_scan(x, dt, A, B, C)
        
        # Skip connection with D
        y = y + self.D.unsqueeze(0).unsqueeze(1) * x
        
        # Gating
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        
        # Residual connection
        output = output + residual
        
        # Post-normalization (if not using pre-norm)
        if not self.config.use_pre_norm:
            output = self.norm(output)
        
        return output
    
    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Selective scan implementation
        Simplified version - production would use optimized CUDA kernels
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[-1]
        
        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        # Sequential scan
        for t in range(seq_len):
            # Discretize A and B
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)  # (batch, d_inner, d_state)
            dB = dt[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1)  # (batch, d_inner, d_state)
            
            # Update state
            h = h * dA + dB * x[:, t].unsqueeze(-1)  # (batch, d_inner, d_state)
            
            # Output
            y_t = torch.sum(h * C[:, t].unsqueeze(1), dim=-1)  # (batch, d_inner)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
        return y


class ProductionTransformerBlock(nn.Module):
    """
    Enhanced Transformer Block for production
    Includes GQA, SwiGLU, and pre-normalization
    """
    def __init__(self, config: ProductionModelConfig):
        super().__init__()
        self.config = config
        
        # Pre-norm
        self.attention_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Attention
        self.attention = GroupedQueryAttention(config)
        
        # FFN
        if config.use_gated_ffn:
            self.ffn = SwiGLU(config.d_model, config.ffn_dim)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.d_model, config.ffn_dim, bias=False),
                nn.GELU(),
                nn.Linear(config.ffn_dim, config.d_model, bias=False),
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states: (batch, seq_len, d_model)
            attention_mask: (batch, 1, seq_len, seq_len)
            position_ids: (batch, seq_len)
            past_key_value: Cached (key, value) for efficient inference
            use_cache: Whether to return updated cache
        """
        residual = hidden_states
        
        # Self-attention with pre-norm
        hidden_states = self.attention_norm(hidden_states)
        hidden_states, present_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # FFN with pre-norm
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class ProductionHybridModel(nn.Module):
    """
    Production-grade Hybrid Mamba-Transformer Model
    Optimized for both quality and efficiency
    """
    def __init__(self, config: ProductionModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        # Position embeddings (if not using RoPE)
        if not config.use_rotary_embeddings:
            self.embed_positions = nn.Embedding(config.max_seq_length, config.d_model)
        
        # Build layers according to pattern
        self.layers = nn.ModuleList()
        layer_pattern = config.get_layer_pattern()
        
        for layer_type in layer_pattern:
            if layer_type == "mamba":
                self.layers.append(ProductionMambaBlock(config))
            else:
                self.layers.append(ProductionTransformerBlock(config))
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Output head
        if config.tie_word_embeddings:
            self.lm_head = None  # Will share with embed_tokens
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
    
    def _init_weights(self, module):
        """Initialize weights with scaled initialization"""
        std = self.config.initializer_range
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
        # Scaled initialization for deeper layers
        if self.config.use_scaled_init:
            for name, p in module.named_parameters():
                if "out_proj" in name or "o_proj" in name or "w2" in name:
                    # Scale down output projections
                    p.data.normal_(mean=0.0, std=std / math.sqrt(2 * self.config.n_layers))
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Tuple:
        """
        Forward pass of the model
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            position_ids: (batch, seq_len)
            past_key_values: List of (key, value) tuples for each transformer layer
            use_cache: Whether to use KV caching
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return dict or tuple
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_length = input_ids.shape
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Create position IDs if not provided
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Add position embeddings if not using RoPE
        if not self.config.use_rotary_embeddings:
            hidden_states = hidden_states + self.embed_positions(position_ids)
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=hidden_states.device)
        
        # Convert to 4D causal mask
        attention_mask = self._prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values
        )
        
        # Process through layers
        all_hidden_states = () if output_hidden_states else None
        next_cache = [] if use_cache else None
        
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if isinstance(layer, ProductionMambaBlock):
                # Mamba blocks don't use attention mask or cache
                if self.gradient_checkpointing and self.training:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        layer,
                        hidden_states,
                        use_reentrant=False
                    )
                else:
                    hidden_states = layer(hidden_states)
                
                if use_cache:
                    next_cache.append(None)
            
            else:  # ProductionTransformerBlock
                if self.gradient_checkpointing and self.training:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, past_key_value=None, use_cache=False)
                        return custom_forward
                    
                    hidden_states, _ = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                        position_ids,
                        use_reentrant=False
                    )
                else:
                    hidden_states, present_key_value = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                    )
                
                if use_cache:
                    next_cache.append(present_key_value)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Compute logits
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {
                "logits": logits,
                "past_key_values": next_cache,
                "hidden_states": all_hidden_states,
            }
        else:
            return (logits, next_cache, all_hidden_states)
    
    def _prepare_4d_causal_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        """
        Create 4D causal attention mask
        
        Args:
            attention_mask: (batch, seq_len)
            input_shape: (batch, seq_len)
            inputs_embeds: embedded inputs for dtype and device
            past_key_values_length: length of cached keys/values
        
        Returns:
            attention_mask: (batch, 1, tgt_len, src_len)
        """
        batch_size, seq_length = input_shape
        combined_attention_mask = None
        
        # Create causal mask
        if seq_length > 1:
            combined_attention_mask = torch.zeros(
                (seq_length, seq_length + (past_key_values_length if past_key_values_length else 0)),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device
            )
            combined_attention_mask = torch.triu(
                torch.ones_like(combined_attention_mask) * float("-inf"),
                diagonal=1
            )
        
        # Expand to 4D
        if combined_attention_mask is not None:
            combined_attention_mask = combined_attention_mask.unsqueeze(0).unsqueeze(0)
            combined_attention_mask = combined_attention_mask.expand(
                batch_size, 1, seq_length, -1
            )
        
        # Combine with provided attention mask
        if attention_mask is not None and attention_mask.dim() == 2:
            expanded_mask = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)
            expanded_mask = (1.0 - expanded_mask) * float("-inf")
            
            if combined_attention_mask is None:
                combined_attention_mask = expanded_mask
            else:
                combined_attention_mask = combined_attention_mask + expanded_mask
        
        return combined_attention_mask


def create_production_model(model_size: str = "1.3B", device: str = "cuda") -> ProductionHybridModel:
    """
    Create a production model of specified size
    
    Args:
        model_size: One of "1.3B", "2.7B", "6.7B", "13B"
        device: Device to place model on
    
    Returns:
        model: Initialized production model
    """
    from production_config import get_config, print_config_summary
    
    config = get_config(model_size)
    print_config_summary(config)
    
    print(f"\n🔧 Initializing model...")
    model = ProductionHybridModel(config)
    
    # Move to device
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device)
        print(f"✅ Model moved to CUDA")
    else:
        print(f"⚠️  CUDA not available, using CPU")
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing Production Model Creation\n")
    
    # Create small test model
    model = create_production_model("1.3B", device="cpu")
    
    # Test forward pass
    print(f"\n🧪 Testing forward pass...")
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        output = model(input_ids)
    
    print(f"✅ Forward pass successful!")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output['logits'].shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {model.config.vocab_size})")
