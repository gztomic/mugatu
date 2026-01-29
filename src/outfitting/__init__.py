"""
Outfitting - Outfit Recommendation Model

A complete ML pipeline for outfit recommendation using CLIP embeddings and Transformers.
Supports both base outfit completion and style-conditioned recommendations.
"""

__version__ = "0.1.0"

from .training.models import OutfitTransformerMLM
from .training.losses import OutfitLoss, HardNegativeMiner
from .training.trainer import OutfitTrainer, TrainingConfig
from .serving.inference import OutfitCompleter
from .dataprep.dataset import OutfitDataset, collate_outfits

__all__ = [
    "OutfitTransformerMLM",
    "OutfitLoss",
    "HardNegativeMiner",
    "OutfitTrainer",
    "TrainingConfig",
    "OutfitCompleter",
    "OutfitDataset",
    "collate_outfits",
]
