"""
XAI Metrics Framework for Vision Transformers.

Implements:
- Attention Heatmap Extraction
- Attention Entropy (Focus metric)
- Deletion Score (Robustness/Faithfulness metric)
- Insertion Score (Robustness/Faithfulness metric)
- Sparsity Index
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import torch
from torch.nn import functional as F
import numpy as np


class XAIMetric(ABC):
    """Abstract base class for XAI metrics."""

    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        pass


class AttentionHeatmap:
    """Extracts CLS-to-patch attention map from ViT attentions."""

    @staticmethod
    def extract(
        attentions: Tuple[torch.Tensor, ...],
        layer_idx: int = -1,
        average_layers: bool = False
    ) -> torch.Tensor:
        """
        Extracts the attention map focused on the CLS token.

        Args:
            attentions: Tuple of tensors from model output. 
                        Shape: [batch, heads, seq_len, seq_len]
            layer_idx: Index of the layer to extract.
            average_layers: If True, averages across all layers.

        Returns:
            torch.Tensor: Normalized heatmap [B, num_patches].
        """
        if average_layers:
            # Stack layers: [layers, batch, heads, seq, seq]
            attn = torch.stack(attentions)
            attn = attn.mean(dim=2)  # Avg over heads
            attn = attn.mean(dim=0)  # Avg over layers
        else:
            # Select layer: [batch, heads, seq, seq]
            attn = attentions[layer_idx].mean(dim=1)

        # Select CLS token attention (index 0) to patches (indices 1:)
        # Shape: [batch, num_patches]
        heatmap = attn[:, 0, 1:]
        
        # Normalize to probability distribution
        return heatmap / (heatmap.sum(dim=-1, keepdim=True) + 1e-10)


class AttentionEntropy(XAIMetric):
    """
    Measures dispersion of attention (Shannon Entropy).
    Low = Focused, High = Diffused.
    """

    def compute(self, heatmap: torch.Tensor) -> float:
        """
        Args:
            heatmap: Tensor shape [num_patches] or [1, num_patches].
        """
        if heatmap.ndim == 2:
            heatmap = heatmap[0]

        epsilon = 1e-10
        h_safe = heatmap + epsilon

        # Shannon Entropy: -sum(p * log2(p))
        entropy = -torch.sum(h_safe * torch.log2(h_safe))

        # Normalize by log2(N) to scale between 0 and 1
        n_patches = torch.tensor(len(heatmap), device=heatmap.device)
        entropy /= torch.log2(n_patches)

        return float(entropy.item())


class Sparsity(XAIMetric):
    """Measures attention concentration (Gini-like index)."""

    def compute(self, heatmap: torch.Tensor) -> float:
        """
        Args:
            heatmap: Tensor shape [num_patches].
        Returns:
            float: 0 (uniform) to 1 (sparse).
        """
        if heatmap.ndim == 2:
            heatmap = heatmap[0]

        h = heatmap / (heatmap.sum() + 1e-10)
        h_sorted, _ = torch.sort(h)

        n = len(h)
        idx = torch.arange(1, n + 1, device=h.device)

        # Gini coefficient formula approximation
        sparsity = 1 - 2 * torch.sum((n - idx + 0.5) * h_sorted) / n
        return float(sparsity.item())


class DeletionScore(XAIMetric):
    """
    Measures faithfulness by progressively masking important patches.
    Lower AUC = Better (model relies on important features).
    """

    def __init__(self, steps: int = 10, patch_size: int = 14):
        self.steps = steps
        self.patch_size = patch_size

    def compute(
        self,
        model: torch.nn.Module,
        pixel_values: torch.Tensor,
        heatmap: torch.Tensor,
        target_class: int,
    ) -> float:
        """
        Args:
            pixel_values: Input image [1, 3, H, W].
            heatmap: Attention map [num_patches].
            target_class: Class index to track.
        """
        # Sort patches by importance
        sorted_indices = torch.argsort(heatmap, descending=True)
        total_patches = sorted_indices.shape[0]
        step_size = max(1, total_patches // self.steps)

        scores: List[float] = []
        current_pixels = pixel_values.clone()
        
        # Geometry calculations (Robust to non-square images)
        _, _, h_img, w_img = current_pixels.shape
        patches_per_row = w_img // self.patch_size
        
        # Calculate fill value once (dataset mean or image mean)
        fill_value = torch.mean(current_pixels)

        with torch.no_grad():
            # Initial score
            scores.append(self._predict(model, current_pixels, target_class))

            for i in range(0, total_patches, step_size):
                chunk = sorted_indices[i : i + step_size]

                for patch_idx in chunk:
                    # Map 1D index to 2D pixel coordinates
                    row = (patch_idx // patches_per_row) * self.patch_size
                    col = (patch_idx % patches_per_row) * self.patch_size

                    # Mask patch with mean color
                    current_pixels[
                        :, :, row : row + self.patch_size, col : col + self.patch_size
                    ] = fill_value

                scores.append(
                    self._predict(model, current_pixels, target_class)
                )

        # Compute AUC
        x = np.linspace(0, 1, len(scores))
        auc = np.trapz(scores, x)
        
        # Normalize by initial confidence
        return float(auc / max(scores[0], 1e-10))

    @staticmethod
    def _predict(model, x, target_class) -> float:
        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out
        probs = F.softmax(logits, dim=-1)
        return probs[0, target_class].item()


class InsertionScore(XAIMetric):
    """
    Measures faithfulness by progressively inserting important patches.
    Higher AUC = Better.
    """

    def __init__(self, steps: int = 10, patch_size: int = 14):
        self.steps = steps
        self.patch_size = patch_size

    def compute(
        self,
        model: torch.nn.Module,
        pixel_values: torch.Tensor,
        heatmap: torch.Tensor,
        target_class: int,
    ) -> float:
        
        sorted_indices = torch.argsort(heatmap, descending=True)
        total_patches = sorted_indices.shape[0]
        step_size = max(1, total_patches // self.steps)

        scores: List[float] = []
        
        # Start from empty (black/mean) canvas
        baseline = torch.zeros_like(pixel_values)
        current_pixels = baseline.clone()

        _, _, h_img, w_img = pixel_values.shape
        patches_per_row = w_img // self.patch_size

        with torch.no_grad():
            scores.append(self._predict(model, current_pixels, target_class))

            for i in range(0, total_patches, step_size):
                chunk = sorted_indices[i : i + step_size]

                for patch_idx in chunk:
                    row = (patch_idx // patches_per_row) * self.patch_size
                    col = (patch_idx % patches_per_row) * self.patch_size

                    # Insert original pixels
                    current_pixels[
                        :, :, row : row + self.patch_size, col : col + self.patch_size
                    ] = pixel_values[
                        :, :, row : row + self.patch_size, col : col + self.patch_size
                    ]

                scores.append(
                    self._predict(model, current_pixels, target_class)
                )

        x = np.linspace(0, 1, len(scores))
        auc = np.trapz(scores, x)
        return float(auc)

    @staticmethod
    def _predict(model, x, target_class) -> float:
        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out
        probs = F.softmax(logits, dim=-1)
        return probs[0, target_class].item()


class XAIEvaluator:
    """Facade for running multiple metrics at once."""

    def __init__(self):
        self.entropy = AttentionEntropy()
        self.sparsity = Sparsity()
        self.deletion = DeletionScore()
        self.insertion = InsertionScore()

    def evaluate(
        self,
        model: torch.nn.Module,
        pixel_values: torch.Tensor,
        attentions: Tuple[torch.Tensor, ...],
        target_class: int,
        layer_idx: int = -1,
    ) -> Dict[str, float]:
        """
        Runs all configured metrics.
        
        Args:
            model: The ViT model.
            pixel_values: Input image tensor [1, 3, H, W].
            attentions: Tuple of attention tensors.
            target_class: Index of predicted/target class.
        
        Returns:
            Dictionary containing entropy, sparsity, deletion, insertion scores.
        """
        heatmap = AttentionHeatmap.extract(attentions, layer_idx)[0]

        return {
            "entropy": self.entropy.compute(heatmap),
            "sparsity": self.sparsity.compute(heatmap),
            "deletion": self.deletion.compute(
                model, pixel_values, heatmap, target_class
            ),
            "insertion": self.insertion.compute(
                model, pixel_values, heatmap, target_class
            ),
        }