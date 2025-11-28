"""XAI Metrics Framework for Vision Transformers (Batch-Optimized)."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict
import torch
from torch import nn, Tensor
from torch.nn import functional as F

BatchTensor = Tensor
HeatmapTensor = Tensor
ImageTensor = Tensor
AttentionTensor = Tensor


class XAIMetric(ABC):
    @abstractmethod
    def compute(self, *args, **kwargs) -> BatchTensor:
        pass


class AttentionHeatmap:
    @staticmethod
    def extract(
        attentions: Tuple[AttentionTensor, ...],
        layer_idx: int = -1,
    ) -> HeatmapTensor:
        attn = attentions[layer_idx]
        attn = attn.mean(dim=1)
        heatmap = attn[:, 0, 1:]
        return heatmap / (heatmap.sum(dim=-1, keepdim=True) + 1e-10)


class AttentionEntropy(XAIMetric):
    def compute(self, heatmap: HeatmapTensor) -> BatchTensor:
        epsilon = 1e-10
        h_safe = heatmap + epsilon
        entropy = -torch.sum(h_safe * torch.log2(h_safe), dim=-1)
        n_patches = heatmap.shape[1]
        entropy /= torch.log2(torch.tensor(n_patches, device=heatmap.device, dtype=heatmap.dtype))
        return entropy


class Sparsity(XAIMetric):
    def compute(self, heatmap: HeatmapTensor) -> BatchTensor:
        h = heatmap / (heatmap.sum(dim=-1, keepdim=True) + 1e-10)
        h_sorted, _ = torch.sort(h, dim=-1)
        N = h.shape[1]
        idx = torch.arange(1, N + 1, device=h.device, dtype=h.dtype)
        weighted_sum = torch.sum((N - idx + 0.5) * h_sorted, dim=-1)
        return 1 - 2 * weighted_sum / N


class PerturbationMetric(XAIMetric):
    def __init__(self, steps: int = 10, patch_size: int = 14):
        self.steps = steps
        self.patch_size = patch_size

    @staticmethod
    def _predict(
        model: nn.Module,
        x: ImageTensor,
        target_classes: Tensor,
    ) -> BatchTensor:
        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out
        probs = F.softmax(logits, dim=-1)
        return probs[torch.arange(x.shape[0], device=x.device), target_classes]


class DeletionScore(PerturbationMetric):
    def compute(
        self,
        model: nn.Module,
        pixel_values: ImageTensor,
        heatmap: HeatmapTensor,
        target_classes: Tensor,
    ) -> BatchTensor:
        B, C, H, W = pixel_values.shape
        N = heatmap.shape[1]
        step_size = max(1, N // self.steps)
        patches_per_row = W // self.patch_size

        sorted_idx = torch.argsort(heatmap, descending=True)
        fill_value = pixel_values.mean(dim=(2, 3), keepdim=True)

        patch_rows = (torch.arange(N, device=pixel_values.device) // patches_per_row) * self.patch_size
        patch_cols = (torch.arange(N, device=pixel_values.device) % patches_per_row) * self.patch_size

        scores_list = [self._predict(model, pixel_values, target_classes)]

        canvas = pixel_values.clone()
        for step in range(0, N, step_size):
            patch_idx = sorted_idx[:, step:step + step_size]
            for i in range(patch_idx.shape[1]):
                p_idx = patch_idx[:, i]
                rows = patch_rows[p_idx]
                cols = patch_cols[p_idx]
                for b in range(B):
                    r, c = rows[b].item(), cols[b].item()
                    canvas[b, :, r:r + self.patch_size, c:c + self.patch_size] = fill_value[b]
            scores_list.append(self._predict(model, canvas, target_classes))

        scores = torch.stack(scores_list, dim=1)
        x_axis = torch.linspace(0, 1, scores.shape[1], device=pixel_values.device)
        auc = torch.trapezoid(scores, x_axis, dim=1)
        return auc / scores[:, 0].clamp(min=1e-10)


class InsertionScore(PerturbationMetric):
    def compute(
        self,
        model: nn.Module,
        pixel_values: ImageTensor,
        heatmap: HeatmapTensor,
        target_classes: Tensor,
    ) -> BatchTensor:
        B, C, H, W = pixel_values.shape
        N = heatmap.shape[1]
        step_size = max(1, N // self.steps)
        patches_per_row = W // self.patch_size

        sorted_idx = torch.argsort(heatmap, descending=True)

        patch_rows = (torch.arange(N, device=pixel_values.device) // patches_per_row) * self.patch_size
        patch_cols = (torch.arange(N, device=pixel_values.device) % patches_per_row) * self.patch_size

        canvas = torch.zeros_like(pixel_values)
        scores_list = [self._predict(model, canvas, target_classes)]

        for step in range(0, N, step_size):
            patch_idx = sorted_idx[:, step:step + step_size]
            for i in range(patch_idx.shape[1]):
                p_idx = patch_idx[:, i]
                rows = patch_rows[p_idx]
                cols = patch_cols[p_idx]
                for b in range(B):
                    r, c = rows[b].item(), cols[b].item()
                    canvas[b, :, r:r + self.patch_size, c:c + self.patch_size] = \
                        pixel_values[b, :, r:r + self.patch_size, c:c + self.patch_size]
            scores_list.append(self._predict(model, canvas, target_classes))

        scores = torch.stack(scores_list, dim=1)
        x_axis = torch.linspace(0, 1, scores.shape[1], device=pixel_values.device)
        return torch.trapezoid(scores, x_axis, dim=1)


class XAIEvaluator:
    def __init__(self, perturbation_steps: int = 10, patch_size: int = 14):
        self.entropy = AttentionEntropy()
        self.sparsity = Sparsity()
        self.deletion = DeletionScore(steps=perturbation_steps, patch_size=patch_size)
        self.insertion = InsertionScore(steps=perturbation_steps, patch_size=patch_size)

    def evaluate(
        self,
        model: nn.Module,
        pixel_values: ImageTensor,
        attentions: Tuple[AttentionTensor, ...],
        target_classes: Tensor,
        layer_idx: int = -1,
    ) -> Dict[str, BatchTensor]:
        heatmap = AttentionHeatmap.extract(attentions, layer_idx)
        return {
            "entropy": self.entropy.compute(heatmap),
            "sparsity": self.sparsity.compute(heatmap),
            "deletion": self.deletion.compute(model, pixel_values, heatmap, target_classes),
            "insertion": self.insertion.compute(model, pixel_values, heatmap, target_classes),
        }