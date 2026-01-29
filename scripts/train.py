#!/usr/bin/env python
"""
Training script for outfit model.

Usage:
    python scripts/train.py --train_data train.parquet --val_data val.parquet --embeddings embeddings.pt
"""

import argparse
import torch
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from outfitting import train_outfit_model


def main():
    parser = argparse.ArgumentParser(description="Train outfit recommendation model")

    # Data
    parser.add_argument("--train_data", type=str, required=True, help="Path to training parquet")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation parquet")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to item embeddings .pt file")
    parser.add_argument("--category_embeddings", type=str, default=None, help="Path to category embeddings")
    parser.add_argument("--style_embeddings", type=str, default=None, help="Path to style embeddings")

    # Model
    parser.add_argument("--num_categories", type=int, required=True, help="Number of categories")
    parser.add_argument("--num_styles", type=int, default=0, help="Number of styles (0 = no style)")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Transformer hidden dim")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--eval_every_n_steps", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    # Load data
    print(f"Loading training data from {args.train_data}")
    train_df = pd.read_parquet(args.train_data)

    print(f"Loading validation data from {args.val_data}")
    val_df = pd.read_parquet(args.val_data)

    print(f"Loading embeddings from {args.embeddings}")
    embeddings = torch.load(args.embeddings, map_location='cpu', weights_only=True)

    category_embeddings = None
    if args.category_embeddings:
        print(f"Loading category embeddings from {args.category_embeddings}")
        category_embeddings = torch.load(args.category_embeddings, map_location='cpu', weights_only=True)

    style_embeddings = None
    if args.style_embeddings:
        print(f"Loading style embeddings from {args.style_embeddings}")
        style_embeddings = torch.load(args.style_embeddings, map_location='cpu', weights_only=True)

    # Train
    print("\nStarting training...")
    trainer = train_outfit_model(
        train_df=train_df,
        val_df=val_df,
        embeddings=embeddings,
        num_categories=args.num_categories,
        category_embeddings=category_embeddings,
        num_styles=args.num_styles,
        style_embeddings=style_embeddings,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        eval_every_n_steps=args.eval_every_n_steps,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    print(f"\nTraining complete! Best model saved to {args.checkpoint_dir}/best_model.pt")


if __name__ == "__main__":
    main()
