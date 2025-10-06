"""
Production-Grade Hybrid Mamba-Transformer Configuration
Designed for GPT-3 comparable performance with improved efficiency

Model Sizes:
- MAMBA-SLM-1B: 1.3B parameters (comparable to GPT-3 Small/Medium)
- MAMBA-SLM-2.7B: 2.7B parameters (comparable to GPT-3 Large)
- MAMBA-SLM-6.7B: 6.7B parameters (comparable to GPT-3 XL)
- MAMBA-SLM-13B: 13B parameters (comparable to GPT-3 175B with hybrid efficiency)
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import math


@dataclass
class ProductionModelConfig:
    """
    Production model configuration with GPT-3 comparable architecture
    
    Design Philosophy:
    - 65% Mamba blocks for efficient sequence modeling (O(n) complexity)
    - 35% Transformer blocks for powerful attention (strategic placement)
    - Optimized for both quality and inference speed
    """
    
    # Model variant
    model_name: str = "MAMBA-SLM-1.3B"
    
    # Core architecture
    vocab_size: int = 128256  # Llama 3.2 tokenizer (includes multilingual)
    d_model: int = 2048  # Hidden dimension
    n_layers: int = 24  # Total layers
    n_heads: int = 16  # Attention heads
    d_state: int = 16  # Mamba state dimension
    expand_factor: int = 2  # Mamba expansion
    dropout: float = 0.0  # No dropout for large models
    max_seq_length: int = 8192  # Extended context (vs 2048 for GPT-3)
    
    # Hybrid architecture pattern
    mamba_ratio: float = 0.65  # 65% Mamba, 35% Transformer
    transformer_positions: str = "strategic"  # "uniform", "strategic", "top", "bottom"
    
    # Advanced features
    use_flash_attention: bool = True
    use_rotary_embeddings: bool = True  # RoPE for better positional encoding
    use_alibi: bool = False  # ALiBi as alternative to RoPE
    tie_word_embeddings: bool = True  # Share input/output embeddings
    
    # Layer normalization
    layer_norm_eps: float = 1e-5
    use_pre_norm: bool = True  # Pre-LN for better stability
    
    # FFN configuration
    ffn_dim_multiplier: int = 4  # FFN intermediate size = d_model * 4
    use_gated_ffn: bool = True  # SwiGLU activation
    
    # Initialization
    initializer_range: float = 0.02
    use_scaled_init: bool = True  # Scale by depth
    
    # Attention optimizations
    attention_dropout: float = 0.0
    use_grouped_query_attention: bool = True  # GQA for efficiency
    num_key_value_heads: int = 4  # KV heads for GQA
    
    # Memory optimizations
    gradient_checkpointing: bool = True
    use_cache: bool = True  # KV caching for inference
    
    # Training configuration
    max_batch_size: int = 512  # Total across all GPUs
    sequence_parallel: bool = True  # Enable sequence parallelism
    tensor_parallel: bool = True  # Enable tensor parallelism
    pipeline_parallel: bool = False  # Enable for very large models
    
    def __post_init__(self):
        """Validate and compute derived values"""
        # Compute actual layer distribution
        self.n_mamba_layers = int(self.n_layers * self.mamba_ratio)
        self.n_transformer_layers = self.n_layers - self.n_mamba_layers
        
        # Compute FFN dimensions
        self.ffn_dim = self.d_model * self.ffn_dim_multiplier
        
        # Validate head dimensions
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads
        
        # GQA validation
        if self.use_grouped_query_attention:
            assert self.n_heads % self.num_key_value_heads == 0, \
                "n_heads must be divisible by num_key_value_heads"
        
        # Compute total parameters
        self.total_params = self.estimate_parameters()
        
    def estimate_parameters(self) -> int:
        """Estimate total parameter count"""
        params = 0
        
        # Embeddings
        params += self.vocab_size * self.d_model * 2  # Token + position (if not RoPE)
        if self.use_rotary_embeddings:
            params -= self.max_seq_length * self.d_model  # No position embeddings with RoPE
        
        # Mamba blocks
        mamba_params_per_layer = (
            2 * self.d_model * self.d_model * self.expand_factor  # in_proj
            + self.d_model * self.expand_factor * self.d_state * 2  # x_proj
            + self.d_model * self.expand_factor * self.d_state  # dt_proj
            + self.d_model * self.expand_factor * self.d_model  # out_proj
            + self.d_model * self.expand_factor  # conv1d
            + self.d_state  # A_log
            + self.d_model * self.expand_factor  # D
        )
        params += mamba_params_per_layer * self.n_mamba_layers
        
        # Transformer blocks
        if self.use_grouped_query_attention:
            attn_params = (
                self.d_model * self.d_model  # Q projection
                + self.d_model * (self.head_dim * self.num_key_value_heads) * 2  # K, V projections
                + self.d_model * self.d_model  # O projection
            )
        else:
            attn_params = self.d_model * self.d_model * 4  # QKV + O
        
        ffn_params = (
            self.d_model * self.ffn_dim * (3 if self.use_gated_ffn else 2)  # Up + Down (+ Gate)
        )
        
        transformer_params_per_layer = attn_params + ffn_params + self.d_model * 4  # LayerNorms
        params += transformer_params_per_layer * self.n_transformer_layers
        
        # Output head
        if not self.tie_word_embeddings:
            params += self.vocab_size * self.d_model
        
        return params
    
    def get_layer_pattern(self) -> List[str]:
        """Generate layer pattern based on configuration"""
        pattern = []
        
        if self.transformer_positions == "uniform":
            # Evenly distribute transformers
            step = self.n_layers / self.n_transformer_layers
            transformer_indices = {int(i * step) for i in range(self.n_transformer_layers)}
            pattern = ["transformer" if i in transformer_indices else "mamba" 
                      for i in range(self.n_layers)]
        
        elif self.transformer_positions == "strategic":
            # Place transformers at key positions:
            # - Early layers (1-2): Local pattern learning
            # - Middle layers (spaced): Global reasoning
            # - Top layers (last 2-3): Output refinement
            transformer_indices = set()
            
            # Early transformers (first 10%)
            transformer_indices.add(1)
            transformer_indices.add(2)
            
            # Middle transformers (every 4-6 layers)
            middle_start = int(self.n_layers * 0.25)
            middle_end = int(self.n_layers * 0.75)
            for i in range(middle_start, middle_end, 5):
                transformer_indices.add(i)
            
            # Top transformers (last 15%)
            top_start = int(self.n_layers * 0.85)
            for i in range(top_start, self.n_layers):
                transformer_indices.add(i)
            
            # Ensure correct count
            while len(transformer_indices) > self.n_transformer_layers:
                # Remove middle ones first
                middle_positions = [i for i in transformer_indices 
                                   if middle_start <= i < middle_end]
                if middle_positions:
                    transformer_indices.remove(max(middle_positions))
                else:
                    break
            
            while len(transformer_indices) < self.n_transformer_layers:
                # Add more in middle
                for i in range(middle_start, middle_end):
                    if i not in transformer_indices:
                        transformer_indices.add(i)
                        if len(transformer_indices) >= self.n_transformer_layers:
                            break
            
            pattern = ["transformer" if i in transformer_indices else "mamba" 
                      for i in range(self.n_layers)]
        
        elif self.transformer_positions == "top":
            # All transformers at the top
            pattern = ["mamba"] * self.n_mamba_layers + \
                     ["transformer"] * self.n_transformer_layers
        
        elif self.transformer_positions == "bottom":
            # All transformers at the bottom
            pattern = ["transformer"] * self.n_transformer_layers + \
                     ["mamba"] * self.n_mamba_layers
        
        return pattern
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving"""
        return {
            "model_name": self.model_name,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_state": self.d_state,
            "expand_factor": self.expand_factor,
            "dropout": self.dropout,
            "max_seq_length": self.max_seq_length,
            "mamba_ratio": self.mamba_ratio,
            "transformer_positions": self.transformer_positions,
            "use_flash_attention": self.use_flash_attention,
            "use_rotary_embeddings": self.use_rotary_embeddings,
            "use_alibi": self.use_alibi,
            "tie_word_embeddings": self.tie_word_embeddings,
            "layer_norm_eps": self.layer_norm_eps,
            "use_pre_norm": self.use_pre_norm,
            "ffn_dim_multiplier": self.ffn_dim_multiplier,
            "use_gated_ffn": self.use_gated_ffn,
            "use_grouped_query_attention": self.use_grouped_query_attention,
            "num_key_value_heads": self.num_key_value_heads,
            "layer_pattern": self.get_layer_pattern(),
            "total_params": self.total_params,
        }


# Predefined model configurations
PRODUCTION_CONFIGS = {
    "MAMBA-SLM-1.3B": ProductionModelConfig(
        model_name="MAMBA-SLM-1.3B",
        d_model=2048,
        n_layers=24,
        n_heads=16,
        max_seq_length=8192,
        mamba_ratio=0.65,
        transformer_positions="strategic",
    ),
    
    "MAMBA-SLM-2.7B": ProductionModelConfig(
        model_name="MAMBA-SLM-2.7B",
        d_model=2560,
        n_layers=32,
        n_heads=20,
        max_seq_length=8192,
        mamba_ratio=0.65,
        transformer_positions="strategic",
    ),
    
    "MAMBA-SLM-6.7B": ProductionModelConfig(
        model_name="MAMBA-SLM-6.7B",
        d_model=4096,
        n_layers=32,
        n_heads=32,
        num_key_value_heads=8,
        max_seq_length=8192,
        mamba_ratio=0.65,
        transformer_positions="strategic",
    ),
    
    "MAMBA-SLM-13B": ProductionModelConfig(
        model_name="MAMBA-SLM-13B",
        d_model=5120,
        n_layers=40,
        n_heads=40,
        num_key_value_heads=8,
        max_seq_length=8192,
        mamba_ratio=0.65,
        transformer_positions="strategic",
        tensor_parallel=True,
        pipeline_parallel=True,
    ),
}


def get_config(model_size: str = "1.3B") -> ProductionModelConfig:
    """Get production configuration by size"""
    config_name = f"MAMBA-SLM-{model_size}"
    if config_name not in PRODUCTION_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. "
                        f"Available: {list(PRODUCTION_CONFIGS.keys())}")
    return PRODUCTION_CONFIGS[config_name]


def print_config_summary(config: ProductionModelConfig):
    """Print detailed configuration summary"""
    print("=" * 70)
    print(f"Production Model Configuration: {config.model_name}")
    print("=" * 70)
    print(f"\n📊 Model Architecture:")
    print(f"  Total Layers:          {config.n_layers}")
    print(f"  Mamba Layers:          {config.n_mamba_layers} ({config.mamba_ratio*100:.0f}%)")
    print(f"  Transformer Layers:    {config.n_transformer_layers} ({(1-config.mamba_ratio)*100:.0f}%)")
    print(f"  Hidden Dimension:      {config.d_model}")
    print(f"  Attention Heads:       {config.n_heads}")
    if config.use_grouped_query_attention:
        print(f"  KV Heads (GQA):        {config.num_key_value_heads}")
    print(f"  FFN Dimension:         {config.ffn_dim}")
    print(f"  Vocabulary Size:       {config.vocab_size:,}")
    print(f"  Max Sequence Length:   {config.max_seq_length:,}")
    
    print(f"\n💾 Model Size:")
    params_billions = config.total_params / 1e9
    print(f"  Total Parameters:      {config.total_params:,} ({params_billions:.2f}B)")
    fp16_size_gb = (config.total_params * 2) / (1024**3)
    fp32_size_gb = (config.total_params * 4) / (1024**3)
    print(f"  Model Size (FP16):     {fp16_size_gb:.2f} GB")
    print(f"  Model Size (FP32):     {fp32_size_gb:.2f} GB")
    
    print(f"\n⚡ Optimizations:")
    print(f"  Flash Attention:       {'✅' if config.use_flash_attention else '❌'}")
    print(f"  Rotary Embeddings:     {'✅' if config.use_rotary_embeddings else '❌'}")
    print(f"  Grouped Query Attn:    {'✅' if config.use_grouped_query_attention else '❌'}")
    print(f"  Gated FFN (SwiGLU):    {'✅' if config.use_gated_ffn else '❌'}")
    print(f"  Gradient Checkpoint:   {'✅' if config.gradient_checkpointing else '❌'}")
    print(f"  Tied Embeddings:       {'✅' if config.tie_word_embeddings else '❌'}")
    
    print(f"\n🔧 Training Features:")
    print(f"  Max Batch Size:        {config.max_batch_size}")
    print(f"  Tensor Parallel:       {'✅' if config.tensor_parallel else '❌'}")
    print(f"  Sequence Parallel:     {'✅' if config.sequence_parallel else '❌'}")
    print(f"  Pipeline Parallel:     {'✅' if config.pipeline_parallel else '❌'}")
    
    print(f"\n📐 Layer Pattern ({config.transformer_positions}):")
    pattern = config.get_layer_pattern()
    pattern_str = ""
    for i, layer_type in enumerate(pattern):
        symbol = "🔷" if layer_type == "transformer" else "🔶"
        pattern_str += symbol
        if (i + 1) % 20 == 0:
            pattern_str += f"\n  Layer {i-19:2d}-{i:2d}: "
    print(f"  {pattern_str}")
    print(f"\n  Legend: 🔷 Transformer | 🔶 Mamba")
    print("=" * 70)


if __name__ == "__main__":
    # Display all configurations
    for model_name, config in PRODUCTION_CONFIGS.items():
        print_config_summary(config)
        print("\n\n")
