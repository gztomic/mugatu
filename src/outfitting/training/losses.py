"""Loss functions and hard negative mining for outfit training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class OutfitLoss(nn.Module):
    """
    Hybrid loss combining:
    1. MLM loss - fill-in-the-blank reconstruction
    2. Contrastive outfit loss - set-wise ranking (OutfitTransformer style)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        mlm_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        use_hard_negatives: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.mlm_weight = mlm_weight
        self.contrastive_weight = contrastive_weight
        self.use_hard_negatives = use_hard_negatives

    def mlm_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask_positions: torch.Tensor,
        candidate_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MLM loss using cosine similarity in embedding space.

        If candidate_embeddings provided, uses contrastive loss against candidates.
        Otherwise, uses direct cosine similarity loss.
        """
        masked_preds = predictions[mask_positions]
        masked_targets = targets[mask_positions]

        if masked_preds.shape[0] == 0:
            return torch.tensor(0.0, device=predictions.device)

        masked_preds = F.normalize(masked_preds, p=2, dim=-1)
        masked_targets = F.normalize(masked_targets, p=2, dim=-1)

        if candidate_embeddings is not None:
            candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)
            logits = torch.matmul(masked_preds, candidate_embeddings.T) / self.temperature
            target_sims = torch.matmul(masked_targets, candidate_embeddings.T)
            target_indices = target_sims.argmax(dim=-1)
            loss = F.cross_entropy(logits, target_indices)
        else:
            cosine_sim = (masked_preds * masked_targets).sum(dim=-1)
            loss = (1 - cosine_sim).mean()

        return loss

    def contrastive_outfit_loss(
        self,
        outfit_embeddings: torch.Tensor,
        negative_outfit_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Set-wise contrastive loss for outfit embeddings.

        Uses in-batch negatives if negative_outfit_embeddings not provided.
        This is the key insight from OutfitTransformer - treating compatibility
        as a set-level property rather than pairwise item matching.
        """
        batch_size = outfit_embeddings.shape[0]
        device = outfit_embeddings.device

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        outfit_embeddings = F.normalize(outfit_embeddings, p=2, dim=-1)

        if negative_outfit_embeddings is not None:
            negative_outfit_embeddings = F.normalize(negative_outfit_embeddings, p=2, dim=-1)

            pos_scores = torch.ones(batch_size, 1, device=device) / self.temperature

            neg_scores = torch.bmm(
                outfit_embeddings.unsqueeze(1),
                negative_outfit_embeddings.transpose(1, 2)
            ).squeeze(1) / self.temperature

            logits = torch.cat([pos_scores, neg_scores], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)

            loss = F.cross_entropy(logits, labels)
        else:
            sim_matrix = torch.matmul(outfit_embeddings, outfit_embeddings.T) / self.temperature
            labels = torch.arange(batch_size, device=device)
            loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def forward(
        self,
        mlm_predictions: torch.Tensor,
        mlm_targets: torch.Tensor,
        mask_positions: torch.Tensor,
        outfit_embeddings: torch.Tensor,
        negative_outfit_embeddings: Optional[torch.Tensor] = None,
        candidate_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Returns:
            - total_loss: weighted sum of losses
            - loss_dict: individual loss components for logging
        """
        mlm = self.mlm_loss(mlm_predictions, mlm_targets, mask_positions, candidate_embeddings)
        contrastive = self.contrastive_outfit_loss(outfit_embeddings, negative_outfit_embeddings)

        total = self.mlm_weight * mlm + self.contrastive_weight * contrastive

        loss_dict = {
            'total': total.item(),
            'mlm': mlm.item(),
            'contrastive': contrastive.item(),
        }

        return total, loss_dict


class HardNegativeMiner:
    """
    Generate hard negatives for contrastive learning.
    OutfitTransformer-style: create incompatible outfits by strategic item swaps.
    """

    @staticmethod
    def create_swap_negatives(
        item_embeddings: torch.Tensor,
        category_ids: torch.Tensor,
        num_negatives: int = 3,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create hard negatives by swapping items between outfits.

        Strategy: For each outfit, create negatives by replacing one item
        with the corresponding category item from another outfit.
        Only swaps non-padded positions.

        Args:
            item_embeddings: (batch, num_items, embed_dim)
            category_ids: (batch, num_items), -1 for padding
            num_negatives: Number of negative samples per outfit
            padding_mask: (batch, num_items), True = padding

        Returns:
            negative_embeddings: (batch, num_negatives, num_items, embed_dim)
            negative_categories: (batch, num_negatives, num_items)
        """
        batch_size, num_items, embed_dim = item_embeddings.shape
        device = item_embeddings.device

        if padding_mask is None:
            padding_mask = (category_ids == -1)

        negative_embeddings = []
        negative_categories = []

        for _ in range(num_negatives):
            perm = torch.randperm(batch_size, device=device)

            neg_embeds = item_embeddings.clone()
            neg_cats = category_ids.clone()

            for i in range(batch_size):
                valid_positions = (~padding_mask[i]).nonzero(as_tuple=True)[0]
                if len(valid_positions) == 0:
                    continue

                swap_idx = valid_positions[
                    torch.randint(len(valid_positions), (1,), device=device)
                ].item()

                source_valid = (~padding_mask[perm[i]]).nonzero(as_tuple=True)[0]
                if len(source_valid) == 0:
                    continue

                source_idx = source_valid[
                    torch.randint(len(source_valid), (1,), device=device)
                ].item()

                neg_embeds[i, swap_idx] = item_embeddings[perm[i], source_idx]
                neg_cats[i, swap_idx] = category_ids[perm[i], source_idx]

            negative_embeddings.append(neg_embeds)
            negative_categories.append(neg_cats)

        negative_embeddings = torch.stack(negative_embeddings, dim=1)
        negative_categories = torch.stack(negative_categories, dim=1)

        return negative_embeddings, negative_categories


def create_mask_positions(
    batch_size: int,
    num_items: int,
    padding_mask: torch.Tensor,
    mask_prob: float = 0.15,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create random mask positions, ensuring at least one item masked per outfit.

    Args:
        batch_size: Number of outfits
        num_items: Number of items per outfit
        padding_mask: (batch, num_items), True = padding
        mask_prob: Probability of masking each item
        device: Device for tensors

    Returns:
        mask_positions: (batch, num_items), True = masked
    """
    device = device or padding_mask.device

    mask_positions = torch.rand(batch_size, num_items, device=device) < mask_prob
    mask_positions = mask_positions & ~padding_mask

    for i in range(batch_size):
        if not mask_positions[i].any():
            valid = (~padding_mask[i]).nonzero(as_tuple=True)[0]
            if len(valid) > 0:
                rand_idx = valid[torch.randint(len(valid), (1,))].item()
                mask_positions[i, rand_idx] = True

    return mask_positions
