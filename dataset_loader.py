"""
Enhanced Dataset Loader for Hybrid Mamba-Transformer
Supports multiple dataset sources: C4, WikiText, RedPajama, custom files
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDatasetLoader:
    """Unified dataset loader for various text sources"""
    
    SUPPORTED_DATASETS = {
        'wikitext': 'wikitext-103-v1',
        'c4': 'c4',
        'openwebtext': 'openwebtext',
        'bookcorpus': 'bookcorpus',
        'wikipedia': 'wikipedia',
    }
    
    def __init__(self, 
                 tokenizer,
                 max_length: int = 1024,
                 cache_dir: str = "./data_cache",
                 streaming: bool = False):
        """
        Initialize dataset loader
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            cache_dir: Directory for caching datasets
            streaming: Whether to stream large datasets
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.streaming = streaming
        
    def load_from_huggingface(self, 
                              dataset_name: str,
                              split: str = 'train',
                              subset: Optional[str] = None,
                              num_samples: Optional[int] = None) -> Dataset:
        """
        Load dataset from HuggingFace Hub
        
        Args:
            dataset_name: Name of the dataset (e.g., 'wikitext', 'c4')
            split: Dataset split ('train', 'validation', 'test')
            subset: Subset of the dataset (e.g., 'en' for C4)
            num_samples: Limit number of samples (for testing)
            
        Returns:
            Tokenized dataset
        """
        logger.info(f"Loading {dataset_name} dataset from HuggingFace...")
        
        try:
            # Handle different dataset configurations
            if dataset_name == 'wikitext':
                dataset = load_dataset(
                    'wikitext', 
                    'wikitext-103-v1',
                    split=split,
                    cache_dir=str(self.cache_dir),
                    streaming=self.streaming
                )
                text_column = 'text'
                
            elif dataset_name == 'c4':
                dataset = load_dataset(
                    'c4',
                    'en' if subset is None else subset,
                    split=split,
                    cache_dir=str(self.cache_dir),
                    streaming=self.streaming
                )
                text_column = 'text'
                
            elif dataset_name == 'openwebtext':
                dataset = load_dataset(
                    'openwebtext',
                    split=split,
                    cache_dir=str(self.cache_dir),
                    streaming=self.streaming
                )
                text_column = 'text'
                
            elif dataset_name == 'wikipedia':
                dataset = load_dataset(
                    'wikipedia',
                    '20220301.en' if subset is None else subset,
                    split=split,
                    cache_dir=str(self.cache_dir),
                    streaming=self.streaming
                )
                text_column = 'text'
                
            else:
                # Try loading as generic dataset
                dataset = load_dataset(
                    dataset_name,
                    subset,
                    split=split,
                    cache_dir=str(self.cache_dir),
                    streaming=self.streaming
                )
                # Try to detect text column
                text_column = self._detect_text_column(dataset)
            
            # Limit samples if specified
            if num_samples and not self.streaming:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            logger.info(f"Successfully loaded {dataset_name}")
            
            # Create tokenized dataset
            return HuggingFaceTextDataset(
                dataset, 
                self.tokenizer, 
                self.max_length,
                text_column=text_column,
                streaming=self.streaming
            )
            
        except Exception as e:
            logger.error(f"Error loading {dataset_name}: {e}")
            raise
    
    def load_from_files(self, 
                       file_paths: Union[str, List[str]],
                       file_type: str = 'auto') -> Dataset:
        """
        Load dataset from local files
        
        Args:
            file_paths: Single file path or list of file paths
            file_type: File type ('txt', 'json', 'jsonl', 'auto')
            
        Returns:
            Tokenized dataset
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        logger.info(f"Loading {len(file_paths)} file(s)...")
        
        texts = []
        for file_path in file_paths:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            # Detect file type
            if file_type == 'auto':
                file_type = file_path.suffix[1:]  # Remove the dot
            
            # Load based on file type
            if file_type == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            
            elif file_type in ['json', 'jsonl']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_type == 'jsonl':
                        for line in f:
                            data = json.loads(line)
                            texts.append(self._extract_text_from_json(data))
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                texts.append(self._extract_text_from_json(item))
                        else:
                            texts.append(self._extract_text_from_json(data))
        
        logger.info(f"Loaded {len(texts)} texts from files")
        
        return SimpleTextDataset(texts, self.tokenizer, self.max_length)
    
    def load_from_directory(self,
                           directory: str,
                           pattern: str = "*.txt",
                           recursive: bool = True) -> Dataset:
        """
        Load all matching files from a directory
        
        Args:
            directory: Directory path
            pattern: File pattern (e.g., '*.txt', '*.json')
            recursive: Search recursively
            
        Returns:
            Tokenized dataset
        """
        directory = Path(directory)
        
        if recursive:
            file_paths = list(directory.rglob(pattern))
        else:
            file_paths = list(directory.glob(pattern))
        
        logger.info(f"Found {len(file_paths)} files matching '{pattern}'")
        
        return self.load_from_files([str(p) for p in file_paths])
    
    def _detect_text_column(self, dataset) -> str:
        """Try to detect which column contains text"""
        common_names = ['text', 'content', 'body', 'article', 'document']
        
        if hasattr(dataset, 'column_names'):
            columns = dataset.column_names
        elif hasattr(dataset, 'features'):
            columns = list(dataset.features.keys())
        else:
            raise ValueError("Cannot detect columns in dataset")
        
        for name in common_names:
            if name in columns:
                return name
        
        # Default to first column
        logger.warning(f"Could not detect text column, using '{columns[0]}'")
        return columns[0]
    
    def _extract_text_from_json(self, data: Dict) -> str:
        """Extract text from JSON object"""
        # Try common text field names
        for key in ['text', 'content', 'body', 'article', 'document']:
            if key in data:
                return data[key]
        
        # If not found, concatenate all string values
        texts = [v for v in data.values() if isinstance(v, str)]
        return ' '.join(texts) if texts else str(data)


class SimpleTextDataset(Dataset):
    """Simple dataset for list of texts"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


class HuggingFaceTextDataset(Dataset):
    """Dataset wrapper for HuggingFace datasets"""
    
    def __init__(self, 
                 hf_dataset,
                 tokenizer,
                 max_length: int = 1024,
                 text_column: str = 'text',
                 streaming: bool = False):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.streaming = streaming
    
    def __len__(self):
        if self.streaming:
            return float('inf')  # Unknown length for streaming
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.streaming:
            # For streaming, iterate to the index
            item = next(self.dataset)
        else:
            item = self.dataset[idx]
        
        text = item[self.text_column]
        
        # Handle empty or very short texts
        if not text or len(text.strip()) < 10:
            text = " "  # Fallback to space
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


class ConcatenatedDataset(Dataset):
    """
    Concatenate multiple datasets
    Useful for mixing different data sources
    """
    
    def __init__(self, datasets: List[Dataset], weights: Optional[List[float]] = None):
        """
        Args:
            datasets: List of datasets to concatenate
            weights: Sampling weights for each dataset (optional)
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = torch.cumsum(torch.tensor([0] + self.lengths), dim=0)
        self.weights = weights
        
        if weights:
            assert len(weights) == len(datasets), "Weights must match number of datasets"
            self.weights = torch.tensor(weights) / sum(weights)
    
    def __len__(self):
        return sum(self.lengths)
    
    def __getitem__(self, idx):
        if self.weights is not None:
            # Sample dataset based on weights
            dataset_idx = torch.multinomial(self.weights, 1).item()
            # Random sample from selected dataset
            item_idx = torch.randint(0, len(self.datasets[dataset_idx]), (1,)).item()
            return self.datasets[dataset_idx][item_idx]
        else:
            # Sequential access
            dataset_idx = torch.searchsorted(self.cumulative_lengths, idx, right=False) - 1
            item_idx = idx - self.cumulative_lengths[dataset_idx].item()
            return self.datasets[dataset_idx][item_idx]


# Utility functions
def create_dataset(
    source: str,
    tokenizer,
    max_length: int = 1024,
    split: str = 'train',
    cache_dir: str = "./data_cache",
    **kwargs
) -> Dataset:
    """
    Convenience function to create a dataset
    
    Args:
        source: Dataset source ('wikitext', 'c4', file path, or directory)
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        split: Dataset split
        cache_dir: Cache directory
        **kwargs: Additional arguments
        
    Returns:
        Dataset instance
    """
    loader = TextDatasetLoader(tokenizer, max_length, cache_dir)
    
    # Check if it's a HuggingFace dataset
    if source in TextDatasetLoader.SUPPORTED_DATASETS or source.startswith('hf:'):
        dataset_name = source.replace('hf:', '')
        return loader.load_from_huggingface(dataset_name, split=split, **kwargs)
    
    # Check if it's a file or directory
    path = Path(source)
    if path.exists():
        if path.is_dir():
            return loader.load_from_directory(str(path), **kwargs)
        else:
            return loader.load_from_files(str(path), **kwargs)
    
    raise ValueError(f"Unknown dataset source: {source}")


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test loading WikiText
    print("Loading WikiText dataset...")
    dataset = create_dataset(
        'wikitext',
        tokenizer,
        max_length=512,
        split='train',
        num_samples=100  # Just for testing
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample: {dataset[0]}")
