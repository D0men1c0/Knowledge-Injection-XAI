"""
LoRA Adapter Management.

Handles the injection of Low-Rank Adaptation (LoRA) layers into
Vision Transformers using the PEFT library.
"""
from typing import List, Optional
import torch
from peft import LoraConfig, get_peft_model, PeftModel


class LoRAFactory:
    """Factory for creating and managing LoRA adapter configurations."""

    @staticmethod
    def create_config(
        r: int,
        lora_alpha: int = 16,
        target_modules: Optional[List[str]] = None,
        dropout: float = 0.1
    ) -> LoraConfig:
        """
        Creates a LoRA configuration object.

        Args:
            r: Rank of the update matrices (e.g., 4, 8, 32).
            lora_alpha: Scaling factor.
            target_modules: List of module names to apply LoRA to (e.g., ["query", "value"]).
                            If None, defaults to DINOv2 standard ["query", "value"].
            dropout: Dropout probability for LoRA layers.

        Returns:
            LoraConfig object from peft library.
        """
        if target_modules is None:
            # Default target modules for ViT/DINOv2 attention blocks
            target_modules = ["query", "value"]

        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            modules_to_save=None,
        )

    @staticmethod
    def inject_adapter(
        model: torch.nn.Module,
        rank: int
    ) -> PeftModel:
        """
        Wraps a base model with a LoRA adapter of a specific rank.

        Args:
            model: The base VisionTransformer (frozen).
            rank: The rank 'r' for the LoRA adapter.

        Returns:
            PeftModel: The model with trainable LoRA layers injected.
        """
        config = LoRAFactory.create_config(r=rank)
        # Apply LoRA
        # Note: In a real scenario, we would load_adapter() from weights.
        # Here we initialize a new adapter for the experiment loop.
        peft_model = get_peft_model(model, config)
        
        # Ensure base model is frozen, only LoRA is trainable (or active)
        peft_model.print_trainable_parameters()
        
        return peft_model