"""
OutfitTransformer MLM Model with Optional Style Conditioning

Key design decisions:
1. NO positional embeddings - outfits are sets, not sequences
2. YES category/slot embeddings - semantic structure matters (top, bottom, shoes, etc.)
3. [CLS] token for outfit-level representation
4. [STYLE] token for optional style conditioning
5. Hybrid loss: MLM (fill-in-the-blank) + Contrastive (outfit ranking)
6. Category and style embeddings can be initialized from CLIP text for semantic alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class OutfitTransformerMLM(nn.Module):
    """
    Transformer for outfit understanding with:
    - Slot/category embeddings instead of positional embeddings
    - [CLS] token for outfit-level representation
    - [STYLE] token for optional style conditioning
    - MLM head for fill-in-the-blank
    - Outfit embedding for contrastive learning

    Category and style embeddings can be:
    - Learned from scratch (default)
    - Initialized from CLIP text embeddings with learnable residual
    """

    def __init__(
        self,
        item_embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_categories: int = 10,
        dropout: float = 0.1,
        max_outfit_size: int = 8,
        category_embeddings_init: Optional[torch.Tensor] = None,
        use_category_residual: bool = True,
        num_styles: int = 0,
        style_embeddings_init: Optional[torch.Tensor] = None,
        use_style_residual: bool = True,
    ):
        """
        Args:
            item_embed_dim: Dimension of input item embeddings (e.g., CLIP dimension)
            hidden_dim: Transformer hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            num_categories: Number of outfit categories/slots
            dropout: Dropout rate
            max_outfit_size: Maximum items per outfit
            category_embeddings_init: Optional (num_categories, embed_dim) CLIP text embeddings
            use_category_residual: Whether to use learnable residual for category embeddings
            num_styles: Number of styles (0 = no style conditioning)
            style_embeddings_init: Optional (num_styles, embed_dim) CLIP text embeddings
            use_style_residual: Whether to use learnable residual for style embeddings
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_categories = num_categories
        self.num_styles = num_styles
        self.use_category_residual = use_category_residual
        self.use_style_residual = use_style_residual

        # Project CLIP embeddings to transformer dimension
        self.item_projection = nn.Linear(item_embed_dim, hidden_dim)

        # Category embeddings (index 0 = CLS, 1 to num_categories = categories)
        if category_embeddings_init is not None:
            cat_embeds = category_embeddings_init.detach().cpu().float()

            if cat_embeds.shape[-1] != hidden_dim:
                proj = nn.Linear(cat_embeds.shape[-1], hidden_dim, bias=False)
                with torch.no_grad():
                    cat_embeds = proj(cat_embeds)

            cls_embed = torch.zeros(1, hidden_dim)
            init_embeds = torch.cat([cls_embed, cat_embeds], dim=0)

            self.category_embeddings = nn.Embedding.from_pretrained(init_embeds, freeze=False)

            if use_category_residual:
                self.category_residual = nn.Parameter(
                    torch.zeros(num_categories + 1, hidden_dim)
                )
            else:
                self.category_residual = None
        else:
            self.category_embeddings = nn.Embedding(num_categories + 1, hidden_dim)
            self.category_residual = None

        # Style embeddings (optional)
        if num_styles > 0:
            if style_embeddings_init is not None:
                style_embeds = style_embeddings_init.detach().cpu().float()

                if style_embeds.shape[-1] != hidden_dim:
                    proj = nn.Linear(style_embeds.shape[-1], hidden_dim, bias=False)
                    with torch.no_grad():
                        style_embeds = proj(style_embeds)

                self.style_embeddings = nn.Embedding.from_pretrained(
                    style_embeds, freeze=False
                )

                if use_style_residual:
                    self.style_residual = nn.Parameter(torch.zeros(num_styles, hidden_dim))
                else:
                    self.style_residual = None
            else:
                self.style_embeddings = nn.Embedding(num_styles, hidden_dim)
                self.style_residual = None

            self.style_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        else:
            self.style_embeddings = None
            self.style_residual = None
            self.style_token = None

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLM prediction head - predicts in CLIP embedding space
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, item_embed_dim)
        )

        # Outfit embedding projection (from CLS token)
        self.outfit_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.embed_ln = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for name, module in self.named_modules():
            if 'category_embeddings' in name or 'style_embeddings' in name:
                continue
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_category_embeddings(self, category_ids: torch.Tensor) -> torch.Tensor:
        """Get category embeddings with optional residual."""
        embeds = self.category_embeddings(category_ids)
        if self.category_residual is not None:
            embeds = embeds + self.category_residual[category_ids]
        return embeds

    def get_style_embeddings(self, style_ids: torch.Tensor) -> torch.Tensor:
        """Get style embeddings with optional residual."""
        embeds = self.style_embeddings(style_ids)
        if self.style_residual is not None:
            embeds = embeds + self.style_residual[style_ids]
        return embeds

    def forward(
        self,
        item_embeddings: torch.Tensor,
        category_ids: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        style_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            item_embeddings: (batch, num_items, item_embed_dim)
            category_ids: (batch, num_items) - category for each item, -1 for padding
            mask_positions: (batch, num_items) - bool, True = masked
            padding_mask: (batch, num_items) - bool, True = padding
            style_ids: (batch,) - style index for each outfit (optional)

        Returns:
            - mlm_predictions: (batch, num_items, item_embed_dim)
            - outfit_embedding: (batch, hidden_dim)
            - item_hidden_states: (batch, num_items, hidden_dim)
        """
        batch_size, num_items, _ = item_embeddings.shape
        device = item_embeddings.device

        if padding_mask is None:
            padding_mask = (category_ids == -1)

        # Project items to hidden dimension
        hidden = self.item_projection(item_embeddings)

        # Replace masked positions with mask token
        if mask_positions is not None:
            mask_expanded = mask_positions.unsqueeze(-1).expand_as(hidden)
            mask_token_expanded = self.mask_token.expand(batch_size, num_items, -1)
            hidden = torch.where(mask_expanded, mask_token_expanded, hidden)

        # Add category embeddings
        safe_category_ids = category_ids.clone()
        safe_category_ids[padding_mask] = 0
        cat_embeds = self.get_category_embeddings(safe_category_ids + 1)
        hidden = hidden + cat_embeds

        # Zero out padded positions
        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Build sequence: [CLS, (STYLE), item1, item2, ...]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_cat_embed = self.get_category_embeddings(
            torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        )
        cls_tokens = cls_tokens + cls_cat_embed

        cls_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        use_style = (style_ids is not None) and (self.style_embeddings is not None)

        if use_style:
            style_embed = self.get_style_embeddings(style_ids).unsqueeze(1)
            style_tokens = self.style_token.expand(batch_size, -1, -1) + style_embed

            hidden = torch.cat([cls_tokens, style_tokens, hidden], dim=1)

            style_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            transformer_padding_mask = torch.cat([cls_pad, style_pad, padding_mask], dim=1)

            prefix_len = 2
        else:
            hidden = torch.cat([cls_tokens, hidden], dim=1)
            transformer_padding_mask = torch.cat([cls_pad, padding_mask], dim=1)

            prefix_len = 1

        hidden = self.embed_ln(hidden)
        hidden = self.transformer(hidden, src_key_padding_mask=transformer_padding_mask)

        # Extract CLS token for outfit embedding
        cls_hidden = hidden[:, 0, :]
        outfit_embedding = self.outfit_projection(cls_hidden)
        outfit_embedding = F.normalize(outfit_embedding, p=2, dim=-1)

        # Extract item hidden states (skip CLS and optional STYLE)
        item_hidden_states = hidden[:, prefix_len:, :]

        # MLM predictions
        mlm_predictions = self.mlm_head(item_hidden_states)

        return mlm_predictions, outfit_embedding, item_hidden_states

    def get_outfit_embedding(
        self,
        item_embeddings: torch.Tensor,
        category_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        style_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get outfit embedding without MLM (for inference)."""
        _, outfit_embedding, _ = self.forward(
            item_embeddings, category_ids,
            mask_positions=None, padding_mask=padding_mask,
            style_ids=style_ids,
        )
        return outfit_embedding
