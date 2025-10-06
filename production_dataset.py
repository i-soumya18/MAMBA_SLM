"""
Production Dataset Curation Strategy
Combines high-quality sources for pre-training production-grade models
Target: 100B+ tokens for GPT-3 comparable quality
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetSource:
    """Configuration for a single dataset source"""
    name: str
    source_type: str  # "huggingface", "local", "streaming"
    path: str
    subset: Optional[str] = None
    split: str = "train"
    
    # Sampling
    max_samples: int = -1  # -1 for all
    weight: float = 1.0  # Relative weight in mixture
    
    # Filtering
    min_length: int = 50  # Minimum text length
    max_length: int = 1000000  # Maximum text length
    remove_duplicates: bool = True
    
    # Quality filters
    min_quality_score: float = 0.0  # 0-1 quality score
    filter_profanity: bool = True
    filter_personal_info: bool = True
    
    # Text fields
    text_column: str = "text"
    
    # Estimated tokens
    estimated_tokens: int = 0
    
    def __str__(self):
        return f"{self.name} ({self.estimated_tokens/1e9:.1f}B tokens, weight={self.weight})"


# High-quality dataset sources for production training
PRODUCTION_DATASETS = {
    # Web crawl data (diverse, large-scale)
    "c4": DatasetSource(
        name="C4 (Colossal Clean Crawled Corpus)",
        source_type="huggingface",
        path="allenai/c4",
        subset="en",
        split="train",
        estimated_tokens=156_000_000_000,  # 156B tokens
        weight=0.30,  # 30% of training
        text_column="text",
    ),
    
    # High-quality diverse text (books, papers, code, etc.)
    "pile": DatasetSource(
        name="The Pile",
        source_type="huggingface",
        path="EleutherAI/pile",
        split="train",
        estimated_tokens=300_000_000_000,  # 300B tokens
        weight=0.25,  # 25% of training
        text_column="text",
    ),
    
    # Code (critical for reasoning)
    "starcoder": DatasetSource(
        name="StarCoder",
        source_type="huggingface",
        path="bigcode/starcoderdata",
        split="train",
        estimated_tokens=250_000_000_000,  # 250B tokens
        weight=0.15,  # 15% of training (important for reasoning)
        text_column="content",
    ),
    
    # Wikipedia (factual knowledge)
    "wikipedia": DatasetSource(
        name="Wikipedia",
        source_type="huggingface",
        path="wikipedia",
        subset="20220301.en",
        split="train",
        estimated_tokens=3_500_000_000,  # 3.5B tokens
        weight=0.05,  # 5% of training
        text_column="text",
    ),
    
    # Books (long-form reasoning)
    "books3": DatasetSource(
        name="Books3",
        source_type="huggingface",
        path="the_pile_books3",
        split="train",
        estimated_tokens=26_000_000_000,  # 26B tokens
        weight=0.10,  # 10% of training
        text_column="text",
    ),
    
    # Scientific papers (technical knowledge)
    "arxiv": DatasetSource(
        name="ArXiv Papers",
        source_type="huggingface",
        path="arxiv_dataset",
        split="train",
        estimated_tokens=15_000_000_000,  # 15B tokens
        weight=0.05,  # 5% of training
        text_column="text",
    ),
    
    # Conversational (dialogue understanding)
    "openwebtext": DatasetSource(
        name="OpenWebText",
        source_type="huggingface",
        path="openwebtext",
        split="train",
        estimated_tokens=8_000_000_000,  # 8B tokens
        weight=0.05,  # 5% of training
        text_column="text",
    ),
    
    # Stack Exchange (Q&A reasoning)
    "stackexchange": DatasetSource(
        name="Stack Exchange",
        source_type="huggingface",
        path="HuggingFaceH4/stack-exchange-preferences",
        split="train",
        estimated_tokens=5_000_000_000,  # 5B tokens
        weight=0.05,  # 5% of training
        text_column="text",
    ),
}


@dataclass
class CurationConfig:
    """Configuration for dataset curation pipeline"""
    
    # Sources
    datasets: List[DatasetSource] = None
    
    # Target size
    target_tokens: int = 100_000_000_000  # 100B tokens
    
    # Quality filters
    enable_quality_filter: bool = True
    min_quality_score: float = 0.3  # Filter out bottom 30%
    
    # Deduplication
    enable_deduplication: bool = True
    dedup_method: str = "minhash"  # "exact", "minhash", "simhash"
    dedup_threshold: float = 0.8  # Similarity threshold
    
    # Content filters
    filter_profanity: bool = True
    filter_personal_info: bool = True
    filter_code_only: bool = False  # Keep code
    
    # Length filters
    min_chars: int = 100
    max_chars: int = 1_000_000
    
    # Language
    target_language: str = "en"
    min_language_confidence: float = 0.9
    
    # Processing
    num_workers: int = 8
    batch_size: int = 1000
    cache_dir: str = "./cache/datasets"
    output_dir: str = "./data/curated"
    
    # Validation
    validation_ratio: float = 0.01  # 1% for validation
    test_ratio: float = 0.005  # 0.5% for test
    
    # Tokenization
    tokenizer_path: str = "meta-llama/Llama-3.2-1B"
    max_seq_length: int = 2048
    
    def __post_init__(self):
        if self.datasets is None:
            # Use default production dataset mix
            self.datasets = list(PRODUCTION_DATASETS.values())


def create_dataset_mixture(
    datasets: List[DatasetSource],
    target_tokens: int
) -> Dict[str, int]:
    """
    Calculate how many samples to take from each dataset
    based on weights and target token count
    """
    # Calculate total weight
    total_weight = sum(d.weight for d in datasets)
    
    # Calculate tokens per dataset
    mixture = {}
    for dataset in datasets:
        normalized_weight = dataset.weight / total_weight
        target_tokens_for_dataset = int(target_tokens * normalized_weight)
        
        # Estimate samples needed (rough estimate: 500 tokens/sample)
        estimated_samples = target_tokens_for_dataset // 500
        
        mixture[dataset.name] = {
            "weight": dataset.weight,
            "target_tokens": target_tokens_for_dataset,
            "estimated_samples": estimated_samples,
            "source": dataset.path,
        }
    
    return mixture


def print_curation_plan(config: CurationConfig):
    """Print detailed curation plan"""
    print("=" * 80)
    print("PRODUCTION DATASET CURATION PLAN")
    print("=" * 80)
    
    print(f"\n🎯 Target: {config.target_tokens / 1e9:.1f}B tokens")
    print(f"📊 Validation: {config.validation_ratio * 100:.1f}%")
    print(f"📊 Test: {config.test_ratio * 100:.1f}%")
    
    print("\n" + "=" * 80)
    print("DATASET MIXTURE")
    print("=" * 80)
    
    mixture = create_dataset_mixture(config.datasets, config.target_tokens)
    
    total_tokens = 0
    for name, info in mixture.items():
        tokens_b = info["target_tokens"] / 1e9
        samples_m = info["estimated_samples"] / 1e6
        total_tokens += info["target_tokens"]
        
        print(f"\n{name}:")
        print(f"  Source: {info['source']}")
        print(f"  Weight: {info['weight'] * 100:.1f}%")
        print(f"  Target Tokens: {tokens_b:.1f}B")
        print(f"  Estimated Samples: {samples_m:.1f}M")
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {total_tokens / 1e9:.1f}B tokens")
    print("=" * 80)
    
    print("\n🔧 Processing Configuration:")
    print(f"  Quality Filter: {'✅' if config.enable_quality_filter else '❌'}")
    print(f"  Deduplication: {'✅' if config.enable_deduplication else '❌'} ({config.dedup_method})")
    print(f"  Profanity Filter: {'✅' if config.filter_profanity else '❌'}")
    print(f"  PII Filter: {'✅' if config.filter_personal_info else '❌'}")
    print(f"  Length Range: {config.min_chars} - {config.max_chars} chars")
    print(f"  Workers: {config.num_workers}")
    
    print("\n💾 Output:")
    print(f"  Cache: {config.cache_dir}")
    print(f"  Output: {config.output_dir}")
    print(f"  Tokenizer: {config.tokenizer_path}")
    print(f"  Max Seq Length: {config.max_seq_length}")


def estimate_processing_time(config: CurationConfig) -> Dict:
    """
    Estimate time required for dataset curation
    
    Based on empirical measurements:
    - Downloading: ~10MB/s
    - Quality filtering: ~100K tokens/s/worker
    - Deduplication: ~50K tokens/s/worker (MinHash)
    - Tokenization: ~200K tokens/s/worker
    """
    total_tokens = config.target_tokens
    
    # Download time (assuming 20% already cached)
    download_gb = (total_tokens * 6) / 1e9 * 0.8  # 6 bytes/token, 80% needs download
    download_hours = download_gb / (10 * 3600)  # 10 MB/s
    
    # Processing time (bottleneck is deduplication)
    processing_rate = 50_000 * config.num_workers  # tokens/s
    processing_hours = total_tokens / processing_rate / 3600
    
    # Tokenization time
    tokenization_rate = 200_000 * config.num_workers  # tokens/s
    tokenization_hours = total_tokens / tokenization_rate / 3600
    
    total_hours = download_hours + processing_hours + tokenization_hours
    total_days = total_hours / 24
    
    # Storage requirements
    raw_storage_gb = (total_tokens * 6) / 1e9  # Raw text
    tokenized_storage_gb = (total_tokens * 2) / 1e9  # Tokenized (int16)
    total_storage_gb = raw_storage_gb + tokenized_storage_gb
    
    return {
        "download_hours": f"{download_hours:.1f}",
        "processing_hours": f"{processing_hours:.1f}",
        "tokenization_hours": f"{tokenization_hours:.1f}",
        "total_hours": f"{total_hours:.1f}",
        "total_days": f"{total_days:.1f}",
        "raw_storage_gb": f"{raw_storage_gb:.1f}",
        "tokenized_storage_gb": f"{tokenized_storage_gb:.1f}",
        "total_storage_gb": f"{total_storage_gb:.1f}",
    }


# Predefined curation configurations
CURATION_CONFIGS = {
    "quick_test": CurationConfig(
        target_tokens=1_000_000_000,  # 1B tokens
        datasets=[PRODUCTION_DATASETS["wikipedia"]],
        enable_deduplication=False,
        num_workers=4,
    ),
    
    "small_scale": CurationConfig(
        target_tokens=10_000_000_000,  # 10B tokens
        datasets=[
            PRODUCTION_DATASETS["c4"],
            PRODUCTION_DATASETS["wikipedia"],
            PRODUCTION_DATASETS["openwebtext"],
        ],
        num_workers=8,
    ),
    
    "medium_scale": CurationConfig(
        target_tokens=50_000_000_000,  # 50B tokens
        datasets=[
            PRODUCTION_DATASETS["c4"],
            PRODUCTION_DATASETS["pile"],
            PRODUCTION_DATASETS["starcoder"],
            PRODUCTION_DATASETS["wikipedia"],
            PRODUCTION_DATASETS["openwebtext"],
        ],
        num_workers=16,
    ),
    
    "production": CurationConfig(
        target_tokens=100_000_000_000,  # 100B tokens
        datasets=list(PRODUCTION_DATASETS.values()),
        num_workers=32,
    ),
    
    "xlarge": CurationConfig(
        target_tokens=300_000_000_000,  # 300B tokens (GPT-3 scale)
        datasets=list(PRODUCTION_DATASETS.values()),
        num_workers=64,
    ),
}


if __name__ == "__main__":
    print("\n" * 2)
    
    # Show all configurations
    for name, config in CURATION_CONFIGS.items():
        print_curation_plan(config)
        
        print("\n⏱️  Estimated Processing Time:")
        estimates = estimate_processing_time(config)
        for key, value in estimates.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 80)
        print("\n" * 2)
    
    print("💡 Usage:")
    print("  from production_dataset import CURATION_CONFIGS")
    print("  config = CURATION_CONFIGS['production']")
    print("  # Then use config in data pipeline")
    print("=" * 80)
