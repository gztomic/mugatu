"""Dataset classes for outfit training."""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from collections import defaultdict
import pandas as pd
import random


class OutfitDataset(Dataset):
    """
    Dataset for outfit data with optional style labels.

    Args:
        df: DataFrame with columns:
            - 'outfit': list of item indices (ints)
            - 'category': list of category indices (ints)
            - 'style': (optional) int style index
        embeddings: Tensor (num_items, embed_dim) indexed by item indices
        max_outfit_size: Maximum items per outfit (truncates if longer)
        has_style: Whether to use style column
        category_to_items: Optional pre-computed mapping of category -> item indices
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: torch.Tensor,
        max_outfit_size: int = 8,
        has_style: bool = False,
        category_to_items: Optional[Dict[int, List[int]]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.max_outfit_size = max_outfit_size
        self.embed_dim = embeddings.shape[1]
        self.has_style = has_style and ('style' in df.columns)

        # Build category -> item mapping if not provided
        if category_to_items is not None:
            self.category_to_items = {k: list(v) for k, v in category_to_items.items()}
        else:
            self.category_to_items = build_category_to_items(df, max_outfit_size)

        print(f"Loaded {len(self.df)} outfits")
        print(f"Embedding matrix: {embeddings.shape}")
        print(f"Categories: {len(self.category_to_items)}")
        if self.has_style:
            print(f"Styles: {df['style'].nunique()} unique")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        item_indices = row['outfit'][:self.max_outfit_size]
        category_indices = row['category'][:self.max_outfit_size]

        item_indices_tensor = torch.tensor(item_indices, dtype=torch.long)
        embeddings = self.embeddings[item_indices_tensor]

        result = {
            'item_indices': item_indices_tensor,
            'embeddings': embeddings,
            'category_ids': torch.tensor(category_indices, dtype=torch.long),
            'num_items': len(item_indices),
        }

        if self.has_style:
            result['style_id'] = torch.tensor(row['style'], dtype=torch.long)

        return result

    def get_candidates_by_category(
        self,
        category_id: int,
        n: int,
        exclude: set = None
    ) -> List[int]:
        """Get random item indices from a category (for FITB evaluation)."""
        candidates = self.category_to_items.get(category_id, [])
        if exclude:
            candidates = [c for c in candidates if c not in exclude]
        if len(candidates) == 0:
            return []
        return random.sample(candidates, min(n, len(candidates)))


def build_category_to_items(
    df: pd.DataFrame,
    max_outfit_size: int = 8
) -> Dict[int, List[int]]:
    """Build mapping from category_id to list of item indices."""
    category_to_items = defaultdict(set)

    for _, row in df.iterrows():
        items = row['outfit'][:max_outfit_size]
        cats = row['category'][:max_outfit_size]
        for item_idx, cat_idx in zip(items, cats):
            category_to_items[cat_idx].add(item_idx)

    return {k: list(v) for k, v in category_to_items.items()}


def collate_outfits(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate variable-length outfits with padding."""
    max_items = max(item['num_items'] for item in batch)
    batch_size = len(batch)
    embed_dim = batch[0]['embeddings'].shape[-1]
    has_style = 'style_id' in batch[0]

    embeddings = torch.zeros(batch_size, max_items, embed_dim)
    category_ids = torch.full((batch_size, max_items), -1, dtype=torch.long)
    item_indices = torch.full((batch_size, max_items), -1, dtype=torch.long)
    padding_mask = torch.ones(batch_size, max_items, dtype=torch.bool)
    num_items = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item['num_items']
        embeddings[i, :n] = item['embeddings']
        category_ids[i, :n] = item['category_ids']
        item_indices[i, :n] = item['item_indices']
        padding_mask[i, :n] = False
        num_items[i] = n

    result = {
        'embeddings': embeddings,
        'category_ids': category_ids,
        'item_indices': item_indices,
        'padding_mask': padding_mask,
        'num_items': num_items,
    }

    if has_style:
        result['style_ids'] = torch.stack([item['style_id'] for item in batch])

    return result
