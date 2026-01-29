"""Training pipeline for OutfitTransformer MLM."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
import logging
from collections import defaultdict
import random
import os

from .models import OutfitTransformerMLM
from .losses import OutfitLoss, HardNegativeMiner, create_mask_positions
from ..dataprep.dataset import OutfitDataset, collate_outfits


@dataclass
class TrainingConfig:
    """Configuration for outfit model training."""

    # Model architecture
    item_embed_dim: int = 1152
    hidden_dim: int = 768
    num_layers: int = 4
    num_heads: int = 8
    num_categories: int = 61
    num_styles: int = 0  # 0 = no style conditioning
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Loss weights
    mlm_weight: float = 1.0
    contrastive_weight: float = 0.5
    temperature: float = 0.07

    # MLM
    mask_prob: float = 0.15
    num_hard_negatives: int = 3

    # Evaluation
    eval_every_n_steps: int = 500
    num_fitb_candidates: int = 100

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000

    # Misc
    num_workers: int = 0  # 0 for Colab compatibility
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class OutfitEvaluator:
    """Fill-in-the-blank and retrieval evaluation."""

    def __init__(
        self,
        model: OutfitTransformerMLM,
        dataset: OutfitDataset,
        num_candidates: int = 100,
        device: str = "cuda",
    ):
        self.model = model
        self.dataset = dataset
        self.num_candidates = num_candidates
        self.device = device
        self.embeddings = dataset.embeddings.to(device)
        self.has_style = dataset.has_style

    @torch.no_grad()
    def evaluate_fitb(
        self,
        dataloader: DataLoader,
        num_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Fill-in-the-blank evaluation."""
        self.model.eval()

        correct = 0
        total = 0
        mrr_sum = 0.0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="FITB Eval", leave=False)):
            if num_batches and batch_idx >= num_batches:
                break

            embeddings = batch['embeddings'].to(self.device)
            category_ids = batch['category_ids'].to(self.device)
            item_indices = batch['item_indices'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            num_items = batch['num_items']

            style_ids = None
            if self.has_style and 'style_ids' in batch:
                style_ids = batch['style_ids'].to(self.device)

            bs = embeddings.shape[0]

            for i in range(bs):
                n = num_items[i].item()
                if n < 2:
                    continue

                mask_idx = random.randint(0, n - 1)
                correct_item = item_indices[i, mask_idx].item()
                correct_cat = category_ids[i, mask_idx].item()

                mask_pos = torch.zeros(1, embeddings.shape[1], dtype=torch.bool, device=self.device)
                mask_pos[0, mask_idx] = True

                style_id = style_ids[i:i+1] if style_ids is not None else None

                mlm_pred, _, _ = self.model(
                    embeddings[i:i+1],
                    category_ids[i:i+1],
                    mask_pos,
                    padding_mask[i:i+1],
                    style_ids=style_id,
                )

                pred_embed = F.normalize(mlm_pred[0, mask_idx].unsqueeze(0), p=2, dim=-1)

                candidate_indices = self.dataset.get_candidates_by_category(
                    correct_cat,
                    self.num_candidates - 1,
                    exclude={correct_item}
                )
                if len(candidate_indices) == 0:
                    continue

                candidate_indices = [correct_item] + candidate_indices

                cand_embeds = F.normalize(self.embeddings[candidate_indices], p=2, dim=-1)

                sims = torch.matmul(pred_embed, cand_embeds.T).squeeze(0)
                ranks = torch.argsort(sims, descending=True)
                rank = (ranks == 0).nonzero(as_tuple=True)[0].item() + 1

                correct += int(rank == 1)
                total += 1
                mrr_sum += 1.0 / rank

        return {
            'fitb_accuracy': correct / max(total, 1),
            'fitb_mrr': mrr_sum / max(total, 1),
            'fitb_samples': total,
        }

    @torch.no_grad()
    def evaluate_retrieval(
        self,
        dataloader: DataLoader,
        num_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Outfit embedding quality check."""
        self.model.eval()

        all_embeds = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Retrieval Eval", leave=False)):
            if num_batches and batch_idx >= num_batches:
                break

            embeddings = batch['embeddings'].to(self.device)
            category_ids = batch['category_ids'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)

            style_ids = None
            if self.has_style and 'style_ids' in batch:
                style_ids = batch['style_ids'].to(self.device)

            outfit_embeds = self.model.get_outfit_embedding(
                embeddings, category_ids, padding_mask, style_ids=style_ids
            )
            all_embeds.append(outfit_embeds.cpu())

        all_embeds = F.normalize(torch.cat(all_embeds, dim=0), p=2, dim=-1)
        sim_matrix = torch.matmul(all_embeds, all_embeds.T)

        self_acc = (sim_matrix.argmax(dim=1) == torch.arange(len(sim_matrix))).float().mean().item()

        mask = ~torch.eye(len(sim_matrix), dtype=torch.bool)
        avg_sim = sim_matrix[mask].mean().item()

        return {
            'retrieval_self_acc': self_acc,
            'retrieval_avg_sim': avg_sim,
            'num_outfits': len(all_embeds),
        }


class OutfitTrainer:
    """Complete training loop with style support."""

    def __init__(
        self,
        config: TrainingConfig,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        embeddings: torch.Tensor,
        category_embeddings_init: Optional[torch.Tensor] = None,
        style_embeddings_init: Optional[torch.Tensor] = None,
        category_to_items: Optional[Dict] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.has_style = config.num_styles > 0

        self._set_seed(config.seed)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Data
        self.train_dataset = OutfitDataset(
            train_df, embeddings,
            has_style=self.has_style,
            category_to_items=category_to_items
        )
        self.val_dataset = OutfitDataset(
            val_df, embeddings,
            has_style=self.has_style,
            category_to_items=category_to_items
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_outfits,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_outfits,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        # Model
        self.model = OutfitTransformerMLM(
            item_embed_dim=config.item_embed_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_categories=config.num_categories,
            dropout=config.dropout,
            category_embeddings_init=category_embeddings_init,
            use_category_residual=True,
            num_styles=config.num_styles,
            style_embeddings_init=style_embeddings_init,
            use_style_residual=True,
        ).to(self.device)

        self.loss_fn = OutfitLoss(
            temperature=config.temperature,
            mlm_weight=config.mlm_weight,
            contrastive_weight=config.contrastive_weight,
        )

        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model parameters: {num_params:,}")
        if self.has_style:
            self.logger.info(f"Style conditioning enabled: {config.num_styles} styles")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = len(self.train_loader) * config.num_epochs

        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / max(1, config.warmup_steps)
            progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Evaluator
        self.evaluator = OutfitEvaluator(
            self.model,
            self.val_dataset,
            num_candidates=config.num_fitb_candidates,
            device=config.device,
        )

        os.makedirs(config.checkpoint_dir, exist_ok=True)

        self.global_step = 0
        self.best_fitb_acc = 0.0
        self.history = defaultdict(list)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()

        embeddings = batch['embeddings'].to(self.device)
        category_ids = batch['category_ids'].to(self.device)
        padding_mask = batch['padding_mask'].to(self.device)

        style_ids = None
        if self.has_style and 'style_ids' in batch:
            style_ids = batch['style_ids'].to(self.device)

        bs, max_items, _ = embeddings.shape

        mask_positions = create_mask_positions(
            bs, max_items, padding_mask,
            self.config.mask_prob, self.device
        )

        neg_embeds, neg_cats = HardNegativeMiner.create_swap_negatives(
            embeddings, category_ids, self.config.num_hard_negatives,
            padding_mask=padding_mask
        )

        # Forward
        mlm_pred, outfit_embed, _ = self.model(
            embeddings, category_ids, mask_positions, padding_mask,
            style_ids=style_ids,
        )

        # Negative outfit embeddings
        b, n_neg, n_items, d = neg_embeds.shape
        neg_flat = neg_embeds.view(b * n_neg, n_items, d)
        neg_cats_flat = neg_cats.view(b * n_neg, n_items)
        pad_expanded = padding_mask.unsqueeze(1).expand(-1, n_neg, -1).reshape(b * n_neg, n_items)

        style_ids_expanded = None
        if style_ids is not None:
            style_ids_expanded = style_ids.unsqueeze(1).expand(-1, n_neg).reshape(b * n_neg)

        with torch.no_grad():
            _, neg_outfit_embeds, _ = self.model(
                neg_flat, neg_cats_flat, padding_mask=pad_expanded,
                style_ids=style_ids_expanded,
            )
        neg_outfit_embeds = neg_outfit_embeds.view(b, n_neg, -1)

        # Loss
        total_loss, loss_dict = self.loss_fn(
            mlm_predictions=mlm_pred,
            mlm_targets=embeddings,
            mask_positions=mask_positions,
            outfit_embeddings=outfit_embed,
            negative_outfit_embeddings=neg_outfit_embeds,
        )

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        loss_dict['lr'] = self.scheduler.get_last_lr()[0]
        return loss_dict

    def evaluate(self, num_batches: int = 50) -> Dict[str, float]:
        self.logger.info("Evaluating...")
        fitb = self.evaluator.evaluate_fitb(self.val_loader, num_batches=num_batches)
        retrieval = self.evaluator.evaluate_retrieval(self.val_loader, num_batches=20)
        return {**fitb, **retrieval}

    def save_checkpoint(self, name: str):
        path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_fitb_acc': self.best_fitb_acc,
            'config': self.config,
        }, path)
        self.logger.info(f"Saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.global_step = ckpt['global_step']
        self.best_fitb_acc = ckpt.get('best_fitb_acc', 0.0)
        self.logger.info(f"Loaded: {path}")

    def train(self):
        self.logger.info(f"Training for {self.config.num_epochs} epochs")
        self.logger.info(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        self.logger.info(f"Eval every {self.config.eval_every_n_steps} steps")

        for epoch in range(self.config.num_epochs):
            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            tqdm.write(f"{'='*60}")

            epoch_losses = defaultdict(list)
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

            for batch in pbar:
                loss_dict = self.train_step(batch)
                self.global_step += 1

                for k, v in loss_dict.items():
                    epoch_losses[k].append(v)

                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'mlm': f"{loss_dict['mlm']:.4f}",
                    'ctr': f"{loss_dict['contrastive']:.4f}",
                })

                # Periodic eval
                if self.global_step % self.config.eval_every_n_steps == 0:
                    pbar.clear()
                    metrics = self.evaluate()

                    tqdm.write(f"\nStep {self.global_step}:")
                    tqdm.write(f"  FITB Acc: {metrics['fitb_accuracy']:.4f}")
                    tqdm.write(f"  FITB MRR: {metrics['fitb_mrr']:.4f}")

                    for k, v in metrics.items():
                        self.history[f'val_{k}'].append((self.global_step, v))

                    if metrics['fitb_accuracy'] > self.best_fitb_acc:
                        self.best_fitb_acc = metrics['fitb_accuracy']
                        self.save_checkpoint("best_model")
                        tqdm.write(f"  New best: {self.best_fitb_acc:.4f}")

                    pbar.refresh()

                # Periodic save
                if self.global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

            # Epoch summary
            avg = {k: np.mean(v) for k, v in epoch_losses.items()}
            tqdm.write(f"\nEpoch {epoch+1} avg loss: {avg['total']:.4f} "
                      f"(mlm: {avg['mlm']:.4f}, ctr: {avg['contrastive']:.4f})")

            for k, v in avg.items():
                self.history[f'train_{k}'].append((epoch, v))

        # Final eval with best model
        tqdm.write("\n" + "="*60)
        tqdm.write("Final Evaluation (best model)")
        best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            self.load_checkpoint(best_path)

        final = self.evaluate(num_batches=None)
        tqdm.write(f"Final FITB Acc: {final['fitb_accuracy']:.4f}")
        tqdm.write(f"Final FITB MRR: {final['fitb_mrr']:.4f}")

        return final


def train_outfit_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    embeddings: torch.Tensor,
    num_categories: int,
    category_embeddings: Optional[torch.Tensor] = None,
    num_styles: int = 0,
    style_embeddings: Optional[torch.Tensor] = None,
    batch_size: int = 64,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    eval_every_n_steps: int = 500,
    hidden_dim: int = 512,
    num_layers: int = 4,
    mlm_weight: float = 1.0,
    contrastive_weight: float = 0.5,
    checkpoint_dir: str = "checkpoints",
    device: str = None,
    category_to_items: Optional[Dict] = None,
) -> OutfitTrainer:
    """
    Convenience function to train the outfit model.

    Args:
        train_df: DataFrame with 'outfit', 'category', and optionally 'style' columns
        val_df: Same format for validation
        embeddings: Tensor (num_items, embed_dim) - CLIP embeddings
        num_categories: Number of unique categories
        category_embeddings: Optional (num_categories, embed_dim) - CLIP text embeddings
        num_styles: Number of styles (0 = no style conditioning)
        style_embeddings: Optional (num_styles, embed_dim) - CLIP text embeddings for styles
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        eval_every_n_steps: Evaluation frequency
        hidden_dim: Transformer hidden dimension
        num_layers: Number of transformer layers
        mlm_weight: Weight for MLM loss
        contrastive_weight: Weight for contrastive loss
        checkpoint_dir: Directory for saving checkpoints
        device: Device to train on
        category_to_items: Optional pre-computed category to items mapping

    Returns:
        OutfitTrainer with trained model

    Example without styles:
        trainer = train_outfit_model(
            train_df, val_df, embeddings,
            num_categories=57,
            category_embeddings=category_embeddings,
        )

    Example with styles:
        train_df['style'] = style_model.predict(train_df)

        trainer = train_outfit_model(
            train_df, val_df, embeddings,
            num_categories=57,
            category_embeddings=category_embeddings,
            num_styles=10,
            style_embeddings=style_embeddings,
        )
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure embeddings on CPU for DataLoader
    if embeddings.device.type != 'cpu':
        print(f"Moving item embeddings to CPU...")
        embeddings = embeddings.cpu()

    if category_embeddings is not None and category_embeddings.device.type != 'cpu':
        print(f"Moving category embeddings to CPU...")
        category_embeddings = category_embeddings.cpu()

    if style_embeddings is not None and style_embeddings.device.type != 'cpu':
        print(f"Moving style embeddings to CPU...")
        style_embeddings = style_embeddings.cpu()

    config = TrainingConfig(
        item_embed_dim=embeddings.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_categories=num_categories,
        num_styles=num_styles,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        eval_every_n_steps=eval_every_n_steps,
        mlm_weight=mlm_weight,
        contrastive_weight=contrastive_weight,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    trainer = OutfitTrainer(
        config, train_df, val_df, embeddings,
        category_embeddings_init=category_embeddings,
        style_embeddings_init=style_embeddings,
        category_to_items=category_to_items,
    )
    trainer.train()

    return trainer
