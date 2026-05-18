# Copyright 2025 Tencent wechat. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sampler module for WeDLM decoding.

This module handles all token sampling and position selection logic,
including:
- Temperature-based token sampling
- Top-p (nucleus) sampling
- Top-k sampling
- Entropy computation for position selection
- Position selection using entropy-based parallel decoding
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


class Sampler:
    """Handles all token sampling and position selection logic for WeDLM decoding.
    
    This class centralizes all sampling-related operations that were previously
    scattered in model_runner.py, providing a cleaner separation of concerns.
    
    Responsibilities:
    - Sample tokens from logits with temperature scaling
    - Apply top-p (nucleus) and top-k filtering
    - Compute entropy for position selection decisions
    - Select which mask positions to fill based on entropy threshold
    """

    def __init__(self):
        """Initialize the Sampler."""
        pass

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy for each position's probability distribution.
        
        Entropy is used to measure the model's uncertainty at each position.
        Lower entropy indicates higher confidence in the prediction.
        
        Args:
            logits: Raw logits from the model, shape [num_positions, vocab_size]
            
        Returns:
            Entropy values for each position, shape [num_positions]
        """
        return torch.distributions.Categorical(logits=logits).entropy()

    def _apply_top_k(
        self,
        logits: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """Apply top-k filtering to logits.
        
        Sets logits of tokens outside top-k to -inf.
        
        Args:
            logits: Raw logits, shape [num_positions, vocab_size]
            top_k: Number of top tokens to keep. 0 means no filtering.
            
        Returns:
            Filtered logits with same shape as input.
        """
        if top_k <= 0:
            return logits
        
        top_k = min(top_k, logits.size(-1))
        # Get the k-th largest value for each position
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits

    def _apply_top_p(
        self,
        logits: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits.
        
        Keeps the smallest set of tokens whose cumulative probability exceeds top_p.
        
        Args:
            logits: Raw logits, shape [num_positions, vocab_size]
            top_p: Cumulative probability threshold. 1.0 means no filtering.
            
        Returns:
            Filtered logits with same shape as input.
        """
        if top_p >= 1.0:
            return logits
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        # Shift right to keep the first token above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits

    def sample_tokens(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens from logits with temperature, top-p, and top-k.
        
        Supports greedy decoding (temperature=0) and sampling with various
        filtering strategies.
        
        The order of operations is: top-k -> top-p -> temperature scaling -> sample
        
        Args:
            logits: Raw logits from the model, shape [num_positions, vocab_size]
            temperature: Sampling temperature. 0 means greedy decoding.
            top_p: Nucleus sampling threshold. 1.0 means no filtering.
            top_k: Top-k sampling parameter. 0 means no filtering.
            
        Returns:
            Tuple of (sampled_ids, greedy_ids):
            - sampled_ids: Token IDs sampled with the specified strategy
            - greedy_ids: Greedy (argmax) token IDs for reference
        """
        # Compute greedy predictions (always useful for comparison)
        probs = F.softmax(logits, dim=-1)
        greedy_ids = probs.argmax(dim=-1)
        
        if temperature == 0:
            # Greedy decoding - ignore top_p and top_k
            return greedy_ids, greedy_ids
        
        # Apply top-k filtering first
        filtered_logits = self._apply_top_k(logits, top_k)
        
        # Apply top-p filtering
        filtered_logits = self._apply_top_p(filtered_logits, top_p)
        
        # Temperature-scaled sampling
        scaled_logits = filtered_logits / temperature
        sampling_probs = F.softmax(scaled_logits, dim=-1)
        sampled_ids = torch.multinomial(sampling_probs, num_samples=1).squeeze(-1)
        
        return sampled_ids, greedy_ids

    def select_positions_to_fill(
        self,
        entropy: torch.Tensor,
        remaining_mask_indices: List[int],
        entropy_threshold: Optional[float],
        pos_penalty_factor: float
    ) -> List[int]:
        """Select mask positions to fill using entropy-based parallel decoding.
        
        This method implements the core WeDLM position selection algorithm:
        1. Compute position-adjusted entropy by adding a distance penalty
        2. If entropy_threshold is set, select all positions below threshold
        3. Otherwise, select only the position with minimum adjusted entropy
        
        The position penalty encourages the model to decode earlier positions first,
        which helps maintain left-to-right coherence in generation.
        
        Args:
            entropy: Entropy values for each mask position, shape [num_mask_positions]
            remaining_mask_indices: Indices of remaining mask positions in the window
            entropy_threshold: Threshold for parallel decoding. If None, uses greedy
                position selection (single position with min entropy).
            pos_penalty_factor: Factor for position-based penalty. Higher values
                penalize positions further from the start more heavily.
            
        Returns:
            List of indices (into remaining_mask_indices) indicating which
            positions should be filled in this step.
        """
        device = entropy.device
        
        # Convert mask indices to tensor for vectorized computation
        mask_indices_tensor = torch.tensor(
            remaining_mask_indices, device=device, dtype=torch.float
        )
        
        # Compute position-based penalty
        # Positions further from the first mask position get higher penalty
        base_pos = mask_indices_tensor[0]
        distances = mask_indices_tensor - base_pos
        position_penalty = distances * pos_penalty_factor
        
        # Add penalty to entropy
        adjusted_entropy = entropy + position_penalty
        
        # Select positions based on threshold
        if entropy_threshold is not None:
            # Parallel decoding: select all positions with low enough entropy
            candidates = (adjusted_entropy < entropy_threshold).nonzero(as_tuple=True)[0]
            if candidates.numel() > 0:
                return candidates.tolist()
        
        # Fallback: greedy position selection (minimum adjusted entropy)
        return [int(adjusted_entropy.argmin().item())]

    def process_mask_positions(
        self,
        mask_logits: torch.Tensor,
        remaining_mask_indices: List[int],
        temperature: float,
        entropy_threshold: Optional[float],
        pos_penalty_factor: float,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Tuple[List[int], List[int]]:
        """Process mask positions and return positions to fill with their token IDs.
        
        This is the main entry point for sampling during WeDLM decoding. It combines
        all sampling operations: entropy computation, token sampling, and position
        selection.
        
        Args:
            mask_logits: Logits for mask positions only, shape [num_masks, vocab_size]
            remaining_mask_indices: Window indices of remaining mask positions
            temperature: Sampling temperature
            entropy_threshold: Threshold for parallel position selection
            pos_penalty_factor: Position penalty factor for entropy adjustment
            top_p: Nucleus sampling threshold. 1.0 means no filtering.
            top_k: Top-k sampling parameter. 0 means no filtering.
            
        Returns:
            Tuple of (fill_indices, token_ids):
            - fill_indices: Indices into remaining_mask_indices for positions to fill
            - token_ids: Corresponding token IDs to fill at those positions
        """
        if mask_logits.size(0) == 0:
            return [], []
        
        # Step 1: Compute entropy for position selection
        entropy = self.compute_entropy(mask_logits)
        
        # Step 2: Sample tokens with top_p and top_k filtering
        sampled_ids, _ = self.sample_tokens(mask_logits, temperature, top_p, top_k)
        
        # Step 3: Select positions to fill
        fill_indices = self.select_positions_to_fill(
            entropy,
            remaining_mask_indices,
            entropy_threshold,
            pos_penalty_factor
        )
        
        # Step 4: Get token IDs for selected positions
        token_ids = [int(sampled_ids[k].item()) for k in fill_indices]
        
        return fill_indices, token_ids