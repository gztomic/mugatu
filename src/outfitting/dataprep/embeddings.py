"""CLIP embedding utilities for categories and styles."""

import torch
import torch.nn.functional as F
from typing import List


def encode_categories_with_clip(
    category_names: List[str],
    clip_model,
    device: torch.device = None,
    prompt_template: str = "a photo of a {}"
) -> torch.Tensor:
    """
    Encode category names using CLIP text encoder.

    Args:
        category_names: List of category names, e.g. ["top", "bottom", "shoes", "bag"]
        clip_model: Loaded CLIP model (from openai/clip or open_clip)
        device: Device to use
        prompt_template: Template for category names, {} will be replaced with category name

    Returns:
        Tensor of shape (num_categories, embed_dim) with L2-normalized embeddings

    Example:
        import clip
        clip_model, _ = clip.load("ViT-B/32", device="cuda")
        category_names = ["top", "bottom", "shoes", "bag", "accessory"]
        cat_embeds = encode_categories_with_clip(category_names, clip_model)
    """
    try:
        import clip as openai_clip
        tokenize_fn = openai_clip.tokenize
    except ImportError:
        try:
            import open_clip
            tokenize_fn = open_clip.get_tokenizer('ViT-B-32')
        except ImportError:
            raise ImportError(
                "Please install 'clip' (pip install git+https://github.com/openai/CLIP.git) "
                "or 'open_clip' (pip install open_clip_torch)"
            )

    if device is None:
        device = next(clip_model.parameters()).device

    prompts = [prompt_template.format(name) for name in category_names]
    tokens = tokenize_fn(prompts).to(device)

    with torch.no_grad():
        text_embeds = clip_model.encode_text(tokens)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

    return text_embeds.float()


def encode_styles_with_clip(
    style_names: List[str],
    clip_model,
    device: torch.device = None,
    prompt_template: str = "{} style fashion outfit"
) -> torch.Tensor:
    """
    Encode style names using CLIP text encoder.

    Args:
        style_names: List of style names, e.g. ["boho", "minimalist", "romantic"]
        clip_model: Loaded CLIP model
        device: Device to use
        prompt_template: Template for style names

    Returns:
        Tensor of shape (num_styles, embed_dim) with L2-normalized embeddings

    Example:
        style_names = ["boho", "minimalist", "romantic", "streetwear", "maximalist"]
        style_embeds = encode_styles_with_clip(style_names, clip_model)
    """
    return encode_categories_with_clip(
        style_names, clip_model, device, prompt_template
    )
