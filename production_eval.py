"""
Comprehensive Evaluation Suite for Production Models
Includes standard benchmarks: MMLU, HellaSwag, ARC, TruthfulQA
Plus perplexity, generation quality, and inference speed tests
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation"""
    name: str
    dataset_name: str
    dataset_path: str
    subset: Optional[str] = None
    split: str = "test"
    max_samples: int = -1  # -1 for all
    few_shot: int = 0  # Number of few-shot examples
    metric: str = "accuracy"  # accuracy, f1, exact_match, etc.
    
    def __str__(self):
        few_shot_str = f"{self.few_shot}-shot" if self.few_shot > 0 else "0-shot"
        return f"{self.name} ({few_shot_str}, {self.metric})"


# Standard evaluation benchmarks
BENCHMARKS = {
    # MMLU: Massive Multitask Language Understanding
    # Tests knowledge across 57 subjects (STEM, humanities, social sciences)
    "mmlu": BenchmarkConfig(
        name="MMLU (Massive Multitask Language Understanding)",
        dataset_name="mmlu",
        dataset_path="cais/mmlu",
        split="test",
        max_samples=-1,
        few_shot=5,  # 5-shot is standard
        metric="accuracy",
    ),
    
    # HellaSwag: Commonsense reasoning
    # Tests ability to complete common scenarios
    "hellaswag": BenchmarkConfig(
        name="HellaSwag (Commonsense Reasoning)",
        dataset_name="hellaswag",
        dataset_path="Rowan/hellaswag",
        split="validation",
        max_samples=-1,
        few_shot=0,
        metric="accuracy",
    ),
    
    # ARC: AI2 Reasoning Challenge
    # Science questions at grade-school level
    "arc_easy": BenchmarkConfig(
        name="ARC-Easy (Science Questions)",
        dataset_name="arc",
        dataset_path="allenai/ai2_arc",
        subset="ARC-Easy",
        split="test",
        max_samples=-1,
        few_shot=0,
        metric="accuracy",
    ),
    
    "arc_challenge": BenchmarkConfig(
        name="ARC-Challenge (Hard Science Questions)",
        dataset_name="arc",
        dataset_path="allenai/ai2_arc",
        subset="ARC-Challenge",
        split="test",
        max_samples=-1,
        few_shot=25,  # 25-shot is standard for Challenge
        metric="accuracy",
    ),
    
    # TruthfulQA: Tests truthfulness and avoiding misconceptions
    "truthfulqa": BenchmarkConfig(
        name="TruthfulQA (Truthfulness)",
        dataset_name="truthfulqa",
        dataset_path="truthful_qa",
        subset="multiple_choice",
        split="validation",
        max_samples=-1,
        few_shot=0,
        metric="accuracy",
    ),
    
    # PIQA: Physical Interaction QA
    # Tests physical commonsense reasoning
    "piqa": BenchmarkConfig(
        name="PIQA (Physical Commonsense)",
        dataset_name="piqa",
        dataset_path="piqa",
        split="validation",
        max_samples=-1,
        few_shot=0,
        metric="accuracy",
    ),
    
    # WinoGrande: Commonsense reasoning (Winograd Schema)
    "winogrande": BenchmarkConfig(
        name="WinoGrande (Commonsense)",
        dataset_name="winogrande",
        dataset_path="winogrande",
        subset="winogrande_xl",
        split="validation",
        max_samples=-1,
        few_shot=5,
        metric="accuracy",
    ),
    
    # BoolQ: Boolean Questions (Yes/No)
    "boolq": BenchmarkConfig(
        name="BoolQ (Yes/No Questions)",
        dataset_name="boolq",
        dataset_path="boolq",
        split="validation",
        max_samples=-1,
        few_shot=0,
        metric="accuracy",
    ),
    
    # GSM8K: Grade School Math
    # Tests mathematical reasoning
    "gsm8k": BenchmarkConfig(
        name="GSM8K (Math Reasoning)",
        dataset_name="gsm8k",
        dataset_path="gsm8k",
        subset="main",
        split="test",
        max_samples=-1,
        few_shot=8,  # 8-shot CoT is standard
        metric="exact_match",
    ),
    
    # HumanEval: Code generation (Python)
    "humaneval": BenchmarkConfig(
        name="HumanEval (Code Generation)",
        dataset_name="humaneval",
        dataset_path="openai_humaneval",
        split="test",
        max_samples=-1,
        few_shot=0,
        metric="pass@1",
    ),
}


@dataclass
class EvaluationResults:
    """Results from benchmark evaluation"""
    benchmark_name: str
    metric: str
    score: float
    num_samples: int
    num_correct: int = 0
    time_seconds: float = 0.0
    tokens_per_second: float = 0.0
    
    # Detailed results
    per_sample_results: Optional[List[Dict]] = None
    
    def __str__(self):
        return (f"{self.benchmark_name}: {self.score:.2%} "
                f"({self.num_correct}/{self.num_samples})")
    
    def to_dict(self):
        return asdict(self)


class EvaluationSuite:
    """
    Comprehensive evaluation suite for production models
    """
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        
        # Put model in eval mode
        self.model.eval()
        
        # Results storage
        self.results: Dict[str, EvaluationResults] = {}
    
    def evaluate_benchmark(
        self,
        benchmark: BenchmarkConfig,
        max_samples: Optional[int] = None,
    ) -> EvaluationResults:
        """
        Evaluate model on a specific benchmark
        
        Args:
            benchmark: Benchmark configuration
            max_samples: Override max samples (for testing)
        
        Returns:
            EvaluationResults with scores and metrics
        """
        logger.info(f"Evaluating {benchmark.name}...")
        
        # Load dataset
        from datasets import load_dataset
        
        try:
            if benchmark.subset:
                dataset = load_dataset(
                    benchmark.dataset_path,
                    benchmark.subset,
                    split=benchmark.split
                )
            else:
                dataset = load_dataset(
                    benchmark.dataset_path,
                    split=benchmark.split
                )
        except Exception as e:
            logger.error(f"Failed to load {benchmark.name}: {e}")
            return EvaluationResults(
                benchmark_name=benchmark.name,
                metric=benchmark.metric,
                score=0.0,
                num_samples=0,
            )
        
        # Limit samples if specified
        samples = max_samples if max_samples else benchmark.max_samples
        if samples > 0 and samples < len(dataset):
            dataset = dataset.select(range(samples))
        
        # Evaluate based on benchmark type
        if benchmark.dataset_name in ["mmlu", "arc", "hellaswag", "truthfulqa", 
                                       "piqa", "winogrande", "boolq"]:
            results = self._evaluate_multiple_choice(dataset, benchmark)
        elif benchmark.dataset_name == "gsm8k":
            results = self._evaluate_math(dataset, benchmark)
        elif benchmark.dataset_name == "humaneval":
            results = self._evaluate_code(dataset, benchmark)
        else:
            logger.warning(f"Unknown benchmark type: {benchmark.dataset_name}")
            results = EvaluationResults(
                benchmark_name=benchmark.name,
                metric=benchmark.metric,
                score=0.0,
                num_samples=len(dataset),
            )
        
        # Store results
        self.results[benchmark.name] = results
        
        logger.info(f"✓ {results}")
        return results
    
    def _evaluate_multiple_choice(
        self,
        dataset,
        benchmark: BenchmarkConfig,
    ) -> EvaluationResults:
        """Evaluate multiple choice questions"""
        correct = 0
        total = 0
        start_time = time.time()
        total_tokens = 0
        
        for sample in dataset:
            # Format question based on benchmark
            if benchmark.dataset_name == "mmlu":
                question = sample["question"]
                choices = sample["choices"]
                answer = sample["answer"]
            elif benchmark.dataset_name == "hellaswag":
                question = sample["ctx"]
                choices = sample["endings"]
                answer = sample["label"]
            elif benchmark.dataset_name == "arc":
                question = sample["question"]
                choices = sample["choices"]["text"]
                answer = ord(sample["answerKey"]) - ord('A') if len(sample["answerKey"]) == 1 else int(sample["answerKey"])
            elif benchmark.dataset_name == "truthfulqa":
                question = sample["question"]
                choices = sample["mc1_targets"]["choices"]
                answer = sample["mc1_targets"]["labels"].index(1)
            elif benchmark.dataset_name == "piqa":
                question = sample["goal"]
                choices = [sample["sol1"], sample["sol2"]]
                answer = sample["label"]
            elif benchmark.dataset_name == "winogrande":
                question = sample["sentence"]
                choices = [sample["option1"], sample["option2"]]
                answer = int(sample["answer"]) - 1
            elif benchmark.dataset_name == "boolq":
                question = sample["question"]
                choices = ["No", "Yes"]
                answer = int(sample["answer"])
            else:
                continue
            
            # Get model prediction
            prediction = self._get_multiple_choice_answer(question, choices)
            total_tokens += len(self.tokenizer.encode(question))
            
            if prediction == answer:
                correct += 1
            total += 1
        
        elapsed = time.time() - start_time
        
        return EvaluationResults(
            benchmark_name=benchmark.name,
            metric=benchmark.metric,
            score=correct / total if total > 0 else 0.0,
            num_samples=total,
            num_correct=correct,
            time_seconds=elapsed,
            tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0,
        )
    
    def _get_multiple_choice_answer(
        self,
        question: str,
        choices: List[str],
    ) -> int:
        """
        Get model's answer for multiple choice question
        Uses perplexity scoring
        """
        # Score each choice
        scores = []
        for choice in choices:
            text = f"{question}\nAnswer: {choice}"
            score = self._compute_perplexity(text)
            scores.append(score)
        
        # Return choice with lowest perplexity (most likely)
        return int(np.argmin(scores))
    
    def _compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            
            # Compute loss (negative log likelihood)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean"
            )
            
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def _evaluate_math(self, dataset, benchmark: BenchmarkConfig) -> EvaluationResults:
        """Evaluate math reasoning (GSM8K)"""
        # Simplified version - would need proper answer extraction
        logger.warning("Math evaluation not fully implemented - placeholder")
        return EvaluationResults(
            benchmark_name=benchmark.name,
            metric=benchmark.metric,
            score=0.0,
            num_samples=len(dataset),
        )
    
    def _evaluate_code(self, dataset, benchmark: BenchmarkConfig) -> EvaluationResults:
        """Evaluate code generation (HumanEval)"""
        # Would need execution sandbox - placeholder
        logger.warning("Code evaluation not fully implemented - placeholder")
        return EvaluationResults(
            benchmark_name=benchmark.name,
            metric=benchmark.metric,
            score=0.0,
            num_samples=len(dataset),
        )
    
    def evaluate_perplexity(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "test",
    ) -> float:
        """
        Evaluate perplexity on a dataset
        Lower is better
        """
        from datasets import load_dataset
        
        logger.info(f"Evaluating perplexity on {dataset_name}...")
        
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        
        total_loss = 0.0
        total_tokens = 0
        
        for sample in dataset:
            text = sample["text"]
            if len(text.strip()) < 10:
                continue
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.model.config.max_seq_length,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()
                
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum"
                )
                
                total_loss += loss.item()
                total_tokens += shift_labels.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        logger.info(f"✓ Perplexity: {perplexity:.2f}")
        return perplexity
    
    def run_full_evaluation(
        self,
        benchmarks: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, EvaluationResults]:
        """
        Run full evaluation suite
        
        Args:
            benchmarks: List of benchmark names to run (None for all)
            save_path: Path to save results JSON
        
        Returns:
            Dictionary of benchmark results
        """
        if benchmarks is None:
            benchmarks = list(BENCHMARKS.keys())
        
        logger.info(f"Running evaluation on {len(benchmarks)} benchmarks...")
        
        for benchmark_name in benchmarks:
            if benchmark_name in BENCHMARKS:
                self.evaluate_benchmark(BENCHMARKS[benchmark_name])
        
        # Compute average score
        avg_score = np.mean([r.score for r in self.results.values()])
        logger.info(f"\n{'='*70}")
        logger.info(f"Average Score: {avg_score:.2%}")
        logger.info(f"{'='*70}")
        
        # Save results
        if save_path:
            self._save_results(save_path)
        
        return self.results
    
    def _save_results(self, path: str):
        """Save results to JSON"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = {
            name: result.to_dict()
            for name, result in self.results.items()
        }
        
        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {path}")
    
    def print_summary(self):
        """Print evaluation summary"""
        print("=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        for name, result in self.results.items():
            print(f"\n{name}:")
            print(f"  Score: {result.score:.2%}")
            print(f"  Correct: {result.num_correct}/{result.num_samples}")
            if result.time_seconds > 0:
                print(f"  Time: {result.time_seconds:.1f}s")
                print(f"  Speed: {result.tokens_per_second:.1f} tok/s")
        
        avg_score = np.mean([r.score for r in self.results.values()])
        print(f"\n{'='*80}")
        print(f"AVERAGE SCORE: {avg_score:.2%}")
        print(f"{'='*80}")


# GPT-3 baseline scores for comparison
GPT3_BASELINES = {
    "mmlu": 0.437,  # 43.7% (5-shot)
    "hellaswag": 0.788,  # 78.8% (0-shot)
    "arc_easy": 0.683,  # 68.3% (0-shot)
    "arc_challenge": 0.510,  # 51.0% (25-shot)
    "truthfulqa": 0.280,  # 28.0% (0-shot)
    "piqa": 0.811,  # 81.1% (0-shot)
    "winogrande": 0.700,  # 70.0% (5-shot)
    "boolq": 0.760,  # 76.0% (0-shot)
}


if __name__ == "__main__":
    print("=" * 80)
    print("PRODUCTION EVALUATION SUITE")
    print("=" * 80)
    
    print("\nAvailable Benchmarks:")
    for name, config in BENCHMARKS.items():
        print(f"  • {config}")
    
    print("\n" + "=" * 80)
    print("GPT-3 Baseline Scores (for reference):")
    print("=" * 80)
    for name, score in GPT3_BASELINES.items():
        print(f"  {name}: {score:.2%}")
    
    print("\n" + "=" * 80)
    print("💡 Usage:")
    print("  from production_eval import EvaluationSuite")
    print("  evaluator = EvaluationSuite(model, tokenizer)")
    print("  results = evaluator.run_full_evaluation()")
    print("=" * 80)
