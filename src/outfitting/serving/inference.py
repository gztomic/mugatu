"""
Inference utilities for OutfitTransformer MLM with Style Conditioning.

Usage:
    completer = OutfitCompleter(model, item_embeddings, category_to_items)

    # Without style
    results = completer.complete_outfit(
        seed_items=[1234, 5678],
        seed_categories=[0, 1],
        target_categories=[2, 3],
        top_k=10,
    )

    # With style
    results = completer.complete_outfit(
        seed_items=[1234],
        seed_categories=[0],
        target_categories=[1, 2, 3],
        style_id=2,  # e.g., "boho"
        top_k=10,
    )
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import pandas as pd


class OutfitCompleter:
    """Complete outfits given seed items and optional style."""

    def __init__(
        self,
        model,
        item_embeddings: torch.Tensor,
        category_to_items: Dict[int, List[int]],
        device: str = None,
    ):
        """
        Args:
            model: Trained OutfitTransformerMLM
            item_embeddings: (num_items, embed_dim) tensor of CLIP embeddings
            category_to_items: Dict mapping category_id -> list of item indices
            device: Device to run inference on
        """
        self.model = model
        self.model.eval()

        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)

        self.item_embeddings = item_embeddings.cpu()
        self.embed_dim = item_embeddings.shape[1]
        self.item_embeddings_normed = F.normalize(self.item_embeddings, p=2, dim=-1)
        self.category_to_items = category_to_items

        self.has_style = hasattr(model, 'style_embeddings') and model.style_embeddings is not None

    @classmethod
    def from_trainer(cls, trainer):
        """Create from trained OutfitTrainer."""
        return cls(
            model=trainer.model,
            item_embeddings=trainer.train_dataset.embeddings,
            category_to_items=trainer.train_dataset.category_to_items,
            device=trainer.config.device,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        item_embeddings: torch.Tensor,
        category_to_items: Dict[int, List[int]],
        model_kwargs: dict,
        device: str = "cuda",
    ):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint
            item_embeddings: Item embeddings tensor
            category_to_items: Category to items mapping
            model_kwargs: Model init args including:
                - item_embed_dim, hidden_dim, num_layers, num_heads, num_categories
                - category_embeddings_init (required if trained with CLIP init)
                - num_styles, style_embeddings_init (if using style conditioning)
            device: Device to load on
        """
        from ..training.models import OutfitTransformerMLM

        model = OutfitTransformerMLM(**model_kwargs)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        return cls(
            model=model,
            item_embeddings=item_embeddings,
            category_to_items=category_to_items,
            device=device,
        )

    @torch.no_grad()
    def complete_outfit(
        self,
        seed_items: List[int],
        seed_categories: List[int],
        target_categories: List[int],
        style_id: Optional[int] = None,
        top_k: int = 10,
        exclude_seed_items: bool = True,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Complete an outfit given seed items and optional style.

        Args:
            seed_items: List of item indices to start with
            seed_categories: Category for each seed item
            target_categories: Categories to fill
            style_id: Optional style index (e.g., 0=boho, 1=minimal, etc.)
            top_k: Number of candidates per category
            exclude_seed_items: Exclude seed items from results

        Returns:
            Dict mapping target_category -> [(item_idx, score), ...]

        Example:
            results = completer.complete_outfit(
                seed_items=[1234],
                seed_categories=[0],
                target_categories=[1, 2, 3],
                style_id=0,
                top_k=10,
            )
        """
        assert len(seed_items) == len(seed_categories)

        if style_id is not None and not self.has_style:
            print("Warning: style_id provided but model doesn't have style conditioning")
            style_id = None

        # Build sequence
        all_items = list(seed_items) + [0] * len(target_categories)
        all_categories = list(seed_categories) + list(target_categories)
        num_items = len(all_items)

        item_indices = torch.tensor([all_items], dtype=torch.long)
        category_ids = torch.tensor([all_categories], dtype=torch.long)

        # Mask targets
        mask_positions = torch.zeros(1, num_items, dtype=torch.bool)
        mask_positions[0, len(seed_items):] = True

        # Get embeddings
        item_embeds = torch.zeros(1, num_items, self.embed_dim)
        for i, idx in enumerate(seed_items):
            item_embeds[0, i] = self.item_embeddings[idx]

        # Style
        style_ids = None
        if style_id is not None:
            style_ids = torch.tensor([style_id], dtype=torch.long).to(self.device)

        # Forward
        item_embeds = item_embeds.to(self.device)
        category_ids = category_ids.to(self.device)
        mask_positions = mask_positions.to(self.device)

        mlm_pred, _, _ = self.model(
            item_embeds,
            category_ids,
            mask_positions=mask_positions,
            style_ids=style_ids,
        )

        # Find best items per category
        results = {}
        excluded = set(seed_items) if exclude_seed_items else set()

        for i, target_cat in enumerate(target_categories):
            pred_idx = len(seed_items) + i
            pred_embed = mlm_pred[0, pred_idx].cpu()
            pred_embed = F.normalize(pred_embed, dim=-1)

            candidates = self.category_to_items.get(target_cat, [])
            candidates = [c for c in candidates if c not in excluded]

            if not candidates:
                results[target_cat] = []
                continue

            cand_indices = torch.tensor(candidates, dtype=torch.long)
            cand_embeds = self.item_embeddings_normed[cand_indices]

            scores = torch.matmul(cand_embeds, pred_embed)

            k = min(top_k, len(candidates))
            top_scores, top_indices = scores.topk(k)

            results[target_cat] = [
                (candidates[idx.item()], score.item())
                for idx, score in zip(top_indices, top_scores)
            ]

        return results

    @torch.no_grad()
    def complete_outfit_iterative(
        self,
        seed_items: List[int],
        seed_categories: List[int],
        target_categories: List[int],
        style_id: Optional[int] = None,
        top_k: int = 10,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Complete outfit one item at a time (autoregressive).
        Each prediction is added to context before the next.
        """
        current_items = list(seed_items)
        current_categories = list(seed_categories)
        results = {}

        for target_cat in target_categories:
            result = self.complete_outfit(
                seed_items=current_items,
                seed_categories=current_categories,
                target_categories=[target_cat],
                style_id=style_id,
                top_k=top_k,
                exclude_seed_items=True,
            )

            results[target_cat] = result[target_cat]

            if result[target_cat]:
                best_item = result[target_cat][0][0]
                current_items.append(best_item)
                current_categories.append(target_cat)

        return results

    @torch.no_grad()
    def get_outfit_embedding(
        self,
        item_indices: List[int],
        category_ids: List[int],
        style_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Get outfit embedding for a complete outfit."""
        item_indices_t = torch.tensor([item_indices], dtype=torch.long)
        category_ids_t = torch.tensor([category_ids], dtype=torch.long)

        item_embeds = self.item_embeddings[item_indices_t].to(self.device)
        category_ids_t = category_ids_t.to(self.device)

        style_ids = None
        if style_id is not None and self.has_style:
            style_ids = torch.tensor([style_id], dtype=torch.long).to(self.device)

        outfit_embed = self.model.get_outfit_embedding(
            item_embeds, category_ids_t, style_ids=style_ids
        )

        return outfit_embed.squeeze(0).cpu()

    @torch.no_grad()
    def compare_styles(
        self,
        seed_items: List[int],
        seed_categories: List[int],
        target_categories: List[int],
        style_ids: List[int],
        top_k: int = 5,
    ) -> Dict[int, Dict[int, List[Tuple[int, float]]]]:
        """
        Compare outfit completions across multiple styles.

        Returns:
            Dict mapping style_id -> category -> [(item, score), ...]

        Example:
            results = completer.compare_styles(
                seed_items=[1234],
                seed_categories=[0],
                target_categories=[1, 2],
                style_ids=[0, 1, 2],  # boho, minimal, romantic
            )
        """
        results = {}
        for style_id in style_ids:
            results[style_id] = self.complete_outfit(
                seed_items=seed_items,
                seed_categories=seed_categories,
                target_categories=target_categories,
                style_id=style_id,
                top_k=top_k,
            )
        return results

    @torch.no_grad()
    def score_outfit(
        self,
        item_indices: List[int],
        category_ids: List[int],
        style_id: Optional[int] = None,
        method: str = "reconstruction",
    ) -> float:
        """
        Score how good/coherent an outfit is.

        Args:
            item_indices: Item indices in the outfit
            category_ids: Category for each item
            style_id: Optional style
            method: Scoring method
                - "reconstruction": Leave-one-out reconstruction (lower = better)
                - "coherence": Pairwise item similarity (higher = better)

        Returns:
            Score (interpretation depends on method)
        """
        if method == "reconstruction":
            return self._score_reconstruction(item_indices, category_ids, style_id)
        elif method == "coherence":
            return self._score_coherence(item_indices)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _score_reconstruction(
        self,
        item_indices: List[int],
        category_ids: List[int],
        style_id: Optional[int],
    ) -> float:
        """Leave-one-out reconstruction score."""
        num_items = len(item_indices)
        total_loss = 0.0

        style_ids = None
        if style_id is not None and self.has_style:
            style_ids = torch.tensor([style_id], dtype=torch.long).to(self.device)

        for i in range(num_items):
            item_indices_t = torch.tensor([item_indices], dtype=torch.long)
            category_ids_t = torch.tensor([category_ids], dtype=torch.long)

            mask_positions = torch.zeros(1, num_items, dtype=torch.bool)
            mask_positions[0, i] = True

            item_embeds = self.item_embeddings[item_indices_t].to(self.device)
            category_ids_t = category_ids_t.to(self.device)
            mask_positions = mask_positions.to(self.device)

            mlm_pred, _, _ = self.model(
                item_embeds, category_ids_t, mask_positions, style_ids=style_ids
            )

            pred = F.normalize(mlm_pred[0, i], dim=-1)
            actual = F.normalize(self.item_embeddings[item_indices[i]], dim=-1).to(self.device)

            cos_sim = (pred * actual).sum()
            total_loss += (1 - cos_sim).item()

        return total_loss / num_items

    def _score_coherence(self, item_indices: List[int]) -> float:
        """Pairwise item similarity score."""
        if len(item_indices) < 2:
            return 1.0

        embeds = self.item_embeddings_normed[item_indices]
        sim_matrix = torch.matmul(embeds, embeds.T)

        mask = ~torch.eye(len(item_indices), dtype=torch.bool)
        avg_sim = sim_matrix[mask].mean().item()

        return avg_sim


def build_category_to_items(df: pd.DataFrame) -> Dict[int, List[int]]:
    """Build category_to_items from DataFrame."""
    category_to_items = defaultdict(set)

    for _, row in df.iterrows():
        for item_idx, cat_idx in zip(row['outfit'], row['category']):
            category_to_items[cat_idx].add(item_idx)

    return {k: list(v) for k, v in category_to_items.items()}
