"""
Advanced Sampling Strategies for Text Generation
Includes: Beam Search, Contrastive Search, Improved Nucleus Sampling, Streaming
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, List, Dict, Tuple
from dataclasses import dataclass
import math


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length: int = 100
    min_length: int = 0
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    early_stopping: bool = False
    num_return_sequences: int = 1
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int = 1
    
    # Contrastive search specific
    penalty_alpha: float = 0.6
    
    # Streaming
    stream: bool = False


class AdvancedSampler:
    """Advanced sampling methods for text generation"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def generate(self, 
                 input_ids: torch.Tensor,
                 config: GenerationConfig) -> torch.Tensor:
        """
        Main generation method that routes to specific strategies
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            config: Generation configuration
            
        Returns:
            Generated token IDs
        """
        self.model.eval()
        
        # Route to appropriate generation method
        if config.num_beams > 1:
            return self.beam_search(input_ids, config)
        elif config.penalty_alpha > 0 and not config.do_sample:
            return self.contrastive_search(input_ids, config)
        elif config.stream:
            return self.streaming_generate(input_ids, config)
        else:
            return self.nucleus_sampling(input_ids, config)
    
    @torch.no_grad()
    def nucleus_sampling(self,
                        input_ids: torch.Tensor,
                        config: GenerationConfig) -> torch.Tensor:
        """
        Improved nucleus (top-p) sampling with repetition penalty
        
        Args:
            input_ids: Input token IDs
            config: Generation configuration
            
        Returns:
            Generated sequences
        """
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        
        # Track which sequences are done
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        while cur_len < config.max_length:
            # Get model predictions
            outputs = self.model(input_ids)
            next_token_logits = outputs['logits'][:, -1, :]
            
            # Apply temperature
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits,
                    input_ids,
                    config.repetition_penalty
                )
            
            # Apply min length constraint
            if cur_len < config.min_length:
                next_token_logits[:, config.eos_token_id] = -float('inf')
            
            if config.do_sample:
                # Top-k filtering
                if config.top_k > 0:
                    next_token_logits = self._top_k_filtering(next_token_logits, config.top_k)
                
                # Top-p (nucleus) filtering
                if config.top_p < 1.0:
                    next_token_logits = self._top_p_filtering(next_token_logits, config.top_p)
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Update sequences
            next_tokens = next_tokens * unfinished_sequences + config.pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Check for EOS
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.ne(config.eos_token_id).long()
            )
            
            cur_len += 1
            
            # Stop if all sequences are finished
            if unfinished_sequences.max() == 0:
                break
        
        return input_ids
    
    @torch.no_grad()
    def beam_search(self,
                   input_ids: torch.Tensor,
                   config: GenerationConfig) -> torch.Tensor:
        """
        Beam search decoding
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            config: Generation configuration
            
        Returns:
            Generated sequences [batch_size, seq_len]
        """
        batch_size = input_ids.shape[0]
        num_beams = config.num_beams
        
        # Expand input for beam search
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1)
        input_ids = input_ids.reshape(batch_size * num_beams, -1)
        
        # Initialize beam scores
        beam_scores = torch.zeros(batch_size, num_beams, device=self.device)
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially
        beam_scores = beam_scores.view(-1)
        
        cur_len = input_ids.shape[1]
        
        # Track done sequences
        done = [False for _ in range(batch_size)]
        generated_hyps = [
            BeamHypotheses(num_beams, config.max_length, config.length_penalty, early_stopping=config.early_stopping)
            for _ in range(batch_size)
        ]
        
        while cur_len < config.max_length:
            # Get model predictions
            outputs = self.model(input_ids)
            next_token_logits = outputs['logits'][:, -1, :]  # [batch_size * num_beams, vocab_size]
            
            # Apply length penalty
            if config.length_penalty != 1.0:
                next_token_logits = next_token_logits / (cur_len ** config.length_penalty)
            
            # Calculate scores
            vocab_size = next_token_logits.shape[-1]
            next_scores = F.log_softmax(next_token_logits, dim=-1)  # [batch_size * num_beams, vocab_size]
            next_scores = next_scores + beam_scores[:, None]  # Add beam scores
            
            # Reshape for top-k
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            
            # Get top 2*num_beams tokens
            next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
            
            next_batch_beam = []
            
            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # Pad with dummy beams
                    next_batch_beam.extend([(0, config.pad_token_id, 0)] * num_beams)
                    continue
                
                next_sent_beam = []
                
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size
                    
                    effective_beam_id = batch_idx * num_beams + beam_id
                    
                    # Check if this is EOS
                    if token_id.item() == config.eos_token_id:
                        # Add to hypotheses if not too short
                        if cur_len >= config.min_length:
                            generated_hyps[batch_idx].add(
                                input_ids[effective_beam_id].clone(),
                                beam_token_score.item()
                            )
                    else:
                        # Add to next beam
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    
                    if len(next_sent_beam) == num_beams:
                        break
                
                # Check if we're done
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len
                )
                
                next_batch_beam.extend(next_sent_beam)
            
            # Stop if all batches are done
            if all(done):
                break
            
            # Prepare next iteration
            beam_scores = torch.tensor([x[0] for x in next_batch_beam], device=self.device)
            beam_tokens = torch.tensor([x[1] for x in next_batch_beam], device=self.device)
            beam_idx = torch.tensor([x[2] for x in next_batch_beam], device=self.device)
            
            input_ids = input_ids[beam_idx]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(-1)], dim=-1)
            
            cur_len += 1
        
        # Finalize and return best sequences
        best = []
        for batch_idx in range(batch_size):
            # Add remaining beams
            if not done[batch_idx]:
                for beam_id in range(num_beams):
                    effective_beam_id = batch_idx * num_beams + beam_id
                    score = beam_scores[effective_beam_id].item()
                    generated_hyps[batch_idx].add(input_ids[effective_beam_id].clone(), score)
            
            # Select best sequence
            sorted_hyps = sorted(generated_hyps[batch_idx].beams, key=lambda x: x[0], reverse=True)
            best.append(sorted_hyps[0][1])
        
        # Pad to same length
        max_len = max(len(h) for h in best)
        best_padded = torch.full((batch_size, max_len), config.pad_token_id, dtype=torch.long, device=self.device)
        for idx, hyp in enumerate(best):
            best_padded[idx, :len(hyp)] = hyp
        
        return best_padded
    
    @torch.no_grad()
    def contrastive_search(self,
                          input_ids: torch.Tensor,
                          config: GenerationConfig) -> torch.Tensor:
        """
        Contrastive search decoding
        Balances model confidence and degeneration penalty
        
        Args:
            input_ids: Input token IDs
            config: Generation configuration
            
        Returns:
            Generated sequences
        """
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        
        # Store hidden states for similarity calculation
        past_key_values = None
        
        while cur_len < config.max_length:
            # Get model predictions
            outputs = self.model(input_ids)
            next_token_logits = outputs['logits'][:, -1, :]  # [batch_size, vocab_size]
            
            # Apply temperature
            next_token_logits = next_token_logits / config.temperature
            
            # Get top-k candidates
            top_k_logits, top_k_indices = torch.topk(next_token_logits, config.top_k, dim=-1)
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            
            # Calculate degeneration penalty
            if cur_len > 1:
                # Get embeddings for context and candidates
                context_hidden = self.model.embed_tokens(input_ids)  # [batch, seq_len, hidden]
                candidate_hidden = self.model.embed_tokens(top_k_indices)  # [batch, top_k, hidden]
                
                # Calculate cosine similarity
                context_hidden = context_hidden / context_hidden.norm(dim=-1, keepdim=True)
                candidate_hidden = candidate_hidden / candidate_hidden.norm(dim=-1, keepdim=True)
                
                # Max similarity with context
                similarity = torch.matmul(
                    candidate_hidden,  # [batch, top_k, hidden]
                    context_hidden.transpose(1, 2)  # [batch, hidden, seq_len]
                )  # [batch, top_k, seq_len]
                
                degeneration_penalty = similarity.max(dim=-1)[0]  # [batch, top_k]
            else:
                degeneration_penalty = torch.zeros_like(top_k_probs)
            
            # Combine model confidence and degeneration penalty
            scores = (1 - config.penalty_alpha) * top_k_probs - config.penalty_alpha * degeneration_penalty
            
            # Select best token
            best_idx = scores.argmax(dim=-1)
            next_tokens = top_k_indices.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
            
            # Update input
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            cur_len += 1
            
            # Check for EOS
            if (next_tokens == config.eos_token_id).all():
                break
        
        return input_ids
    
    def streaming_generate(self,
                          input_ids: torch.Tensor,
                          config: GenerationConfig,
                          callback: Optional[Callable[[str], None]] = None):
        """
        Streaming generation - yields tokens one at a time
        
        Args:
            input_ids: Input token IDs
            config: Generation configuration
            callback: Function to call with each new token
            
        Yields:
            Generated tokens one at a time
        """
        self.model.eval()
        cur_len = input_ids.shape[1]
        
        with torch.no_grad():
            while cur_len < config.max_length:
                # Get next token
                outputs = self.model(input_ids)
                next_token_logits = outputs['logits'][:, -1, :]
                
                # Apply temperature
                if config.temperature != 1.0:
                    next_token_logits = next_token_logits / config.temperature
                
                # Apply repetition penalty
                if config.repetition_penalty != 1.0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits,
                        input_ids,
                        config.repetition_penalty
                    )
                
                # Top-p sampling
                if config.top_p < 1.0:
                    next_token_logits = self._top_p_filtering(next_token_logits, config.top_p)
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                # Decode and yield
                token_text = self.tokenizer.decode(next_token[0].unsqueeze(0), skip_special_tokens=False)
                
                if callback:
                    callback(token_text)
                
                yield token_text
                
                # Update input
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                cur_len += 1
                
                # Check for EOS
                if next_token[0].item() == config.eos_token_id:
                    break
    
    def _apply_repetition_penalty(self,
                                 logits: torch.Tensor,
                                 input_ids: torch.Tensor,
                                 penalty: float) -> torch.Tensor:
        """Apply repetition penalty to logits"""
        for i in range(input_ids.shape[0]):
            for previous_token in set(input_ids[i].tolist()):
                # Lower probability for repeated tokens
                if logits[i, previous_token] < 0:
                    logits[i, previous_token] *= penalty
                else:
                    logits[i, previous_token] /= penalty
        return logits
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Filter logits to only keep top k values"""
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Filter logits using nucleus (top-p) filtering"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('inf')
        
        return logits


class BeamHypotheses:
    """Helper class for managing beam search hypotheses"""
    
    def __init__(self, num_beams: int, max_length: int, length_penalty: float, early_stopping: bool):
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9
    
    def __len__(self):
        return len(self.beams)
    
    def add(self, hyp: torch.Tensor, sum_logprobs: float):
        """Add a new hypothesis"""
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)
    
    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """Check if we have enough good hypotheses"""
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / (cur_len ** self.length_penalty)
            return self.worst_score >= cur_score


if __name__ == "__main__":
    # Example usage
    print("Advanced sampling module loaded successfully!")
    print("Available methods: nucleus_sampling, beam_search, contrastive_search, streaming_generate")
