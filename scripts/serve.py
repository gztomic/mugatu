#!/usr/bin/env python
"""
Example inference script for outfit completion.

Usage:
    python scripts/serve.py --checkpoint checkpoints/best_model.pt --embeddings embeddings.pt
"""

import argparse
import torch
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from outfitting import OutfitCompleter
from outfitting.dataprep.dataset import build_category_to_items


def main():
    parser = argparse.ArgumentParser(description="Run outfit completion inference")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to item embeddings")
    parser.add_argument("--data", type=str, required=True, help="Path to data parquet (for category_to_items)")
    parser.add_argument("--category_embeddings", type=str, default=None)
    parser.add_argument("--style_embeddings", type=str, default=None)

    parser.add_argument("--num_categories", type=int, required=True)
    parser.add_argument("--num_styles", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    embeddings = torch.load(args.embeddings, map_location='cpu', weights_only=True)
    df = pd.read_parquet(args.data)
    category_to_items = build_category_to_items(df)

    category_embeddings = None
    if args.category_embeddings:
        category_embeddings = torch.load(args.category_embeddings, map_location='cpu', weights_only=True)

    style_embeddings = None
    if args.style_embeddings:
        style_embeddings = torch.load(args.style_embeddings, map_location='cpu', weights_only=True)

    # Build model kwargs
    model_kwargs = {
        'item_embed_dim': embeddings.shape[1],
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'num_categories': args.num_categories,
        'num_styles': args.num_styles,
    }

    if category_embeddings is not None:
        model_kwargs['category_embeddings_init'] = category_embeddings

    if style_embeddings is not None:
        model_kwargs['style_embeddings_init'] = style_embeddings

    # Load model
    print(f"Loading model from {args.checkpoint}")
    completer = OutfitCompleter.from_checkpoint(
        checkpoint_path=args.checkpoint,
        item_embeddings=embeddings,
        category_to_items=category_to_items,
        model_kwargs=model_kwargs,
        device=args.device,
    )

    print("\nModel loaded! Running example completion...")

    # Example: Get a sample outfit from the data
    sample = df.iloc[0]
    seed_items = [sample['outfit'][0]]
    seed_categories = [sample['category'][0]]
    target_categories = list(set(sample['category'][1:]))[:3]

    print(f"\nSeed items: {seed_items}")
    print(f"Seed categories: {seed_categories}")
    print(f"Target categories: {target_categories}")

    results = completer.complete_outfit(
        seed_items=seed_items,
        seed_categories=seed_categories,
        target_categories=target_categories,
        top_k=5,
    )

    print("\nResults:")
    for cat, suggestions in results.items():
        print(f"\n  Category {cat}:")
        for item_idx, score in suggestions:
            print(f"    Item {item_idx}: {score:.4f}")


if __name__ == "__main__":
    main()
