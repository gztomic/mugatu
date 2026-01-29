# Outfitting

A complete ML pipeline for outfit recommendation using CLIP embeddings and Transformers.

## Features

- **Fill-in-the-blank outfit completion**: Given a partial outfit (e.g., a top), recommend items for missing categories (bottoms, shoes, accessories)
- **Style-conditioned recommendations**: Optional style conditioning (boho, minimalist, romantic, etc.) for personalized suggestions
- **Transformer architecture**: Uses a masked language model (MLM) approach with category embeddings
- **CLIP integration**: Leverages CLIP embeddings for semantic understanding of fashion items

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/outfitting.git
cd outfitting

# Install in development mode
pip install -e .

# With CLIP support
pip install -e ".[clip]"
```

## Quick Start

### 1. Prepare Your Data

Your data should be in a pandas DataFrame with:
- `outfit`: list of item indices
- `category`: list of category indices (same length as outfit)
- `style`: (optional) int style index

```python
import pandas as pd

# Example data format
df = pd.DataFrame({
    'outfit': [[0, 1, 2], [3, 4, 5, 6]],      # item indices
    'category': [[0, 1, 2], [0, 1, 2, 3]],    # category indices
    'style': [0, 1],                           # optional style labels
})
```

### 2. Train a Model

```python
import torch
from outfitting import train_outfit_model

# Load your item embeddings (CLIP embeddings)
embeddings = torch.load("item_embeddings.pt")  # (num_items, 1152)

# Optional: Load CLIP text embeddings for categories
category_embeddings = torch.load("category_embeddings.pt")  # (num_categories, 1152)

# Train
trainer = train_outfit_model(
    train_df=train_df,
    val_df=val_df,
    embeddings=embeddings,
    num_categories=57,
    category_embeddings=category_embeddings,
    num_epochs=50,
    batch_size=64,
)
```

### 3. Complete Outfits

```python
from outfitting import OutfitCompleter

# Create completer from trained model
completer = OutfitCompleter.from_trainer(trainer)

# Or load from checkpoint
completer = OutfitCompleter.from_checkpoint(
    checkpoint_path="checkpoints/best_model.pt",
    item_embeddings=embeddings,
    category_to_items=category_to_items,
    model_kwargs={
        'item_embed_dim': 1152,
        'hidden_dim': 512,
        'num_layers': 4,
        'num_heads': 8,
        'num_categories': 57,
    },
    device='cuda',
)

# Complete an outfit
results = completer.complete_outfit(
    seed_items=[1234],           # I have this top
    seed_categories=[0],         # category 0 = top
    target_categories=[1, 2, 3], # Need: bottom, shoes, bag
    top_k=10,
)

for category, suggestions in results.items():
    print(f"Category {category}:")
    for item_idx, score in suggestions[:3]:
        print(f"  Item {item_idx}: {score:.3f}")
```

### 4. Style-Conditioned Completion

```python
# With style conditioning
results = completer.complete_outfit(
    seed_items=[1234],
    seed_categories=[0],
    target_categories=[1, 2, 3],
    style_id=0,  # 0 = boho
    top_k=10,
)

# Compare across styles
style_comparison = completer.compare_styles(
    seed_items=[1234],
    seed_categories=[0],
    target_categories=[1, 2],
    style_ids=[0, 1, 2],  # boho, minimal, romantic
)
```

## Project Structure

```
outfitting/
├── src/outfitting/
│   ├── __init__.py
│   ├── dataprep/
│   │   ├── __init__.py
│   │   ├── dataset.py        # Dataset classes
│   │   └── embeddings.py     # CLIP encoding utilities
│   ├── training/
│   │   ├── __init__.py
│   │   ├── models.py         # OutfitTransformerMLM
│   │   ├── losses.py         # OutfitLoss, HardNegativeMiner
│   │   └── trainer.py        # Training loop
│   └── serving/
│       ├── __init__.py
│       ├── inference.py      # OutfitCompleter
│       └── utils.py          # Visualization utilities
├── scripts/
│   ├── train.py
│   └── serve.py
├── notebooks/
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

## Model Architecture

The model uses a Transformer encoder with:
- **No positional embeddings**: Outfits are sets, not sequences
- **Category embeddings**: Semantic structure (top, bottom, shoes, etc.)
- **[CLS] token**: Outfit-level representation
- **[STYLE] token**: Optional style conditioning
- **MLM head**: Fill-in-the-blank prediction
- **Contrastive loss**: Outfit-level ranking

## Training Details

- **Loss**: Hybrid MLM + Contrastive loss
- **Negatives**: Hard negatives via item swapping between outfits
- **Evaluation**: Fill-in-the-blank (FITB) accuracy and MRR

## API Reference

### `train_outfit_model()`

```python
trainer = train_outfit_model(
    train_df,           # Training DataFrame
    val_df,             # Validation DataFrame
    embeddings,         # Item embeddings (num_items, embed_dim)
    num_categories,     # Number of categories
    category_embeddings=None,  # Optional CLIP text embeddings
    num_styles=0,       # Number of styles (0 = disabled)
    style_embeddings=None,     # Optional style embeddings
    batch_size=64,
    num_epochs=50,
    learning_rate=1e-4,
    hidden_dim=512,
    num_layers=4,
)
```

### `OutfitCompleter`

```python
# Complete outfit
results = completer.complete_outfit(
    seed_items,         # List of item indices
    seed_categories,    # Category for each seed item
    target_categories,  # Categories to fill
    style_id=None,      # Optional style
    top_k=10,
)

# Iterative completion (autoregressive)
results = completer.complete_outfit_iterative(...)

# Get outfit embedding
embedding = completer.get_outfit_embedding(items, categories, style_id)

# Score outfit coherence
score = completer.score_outfit(items, categories, method="reconstruction")
```

## License

MIT
