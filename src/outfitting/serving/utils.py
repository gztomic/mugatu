"""Utility functions for visualization and image handling."""

from PIL import Image
from typing import List, Optional, Tuple


def create_sprite_from_paths(
    paths: List[str],
    images_per_row: Optional[int] = None,
    thumbnail_size: Tuple[int, int] = (224, 224),
    background_color: Tuple[int, int, int] = (255, 255, 255),
    padding: int = 4,
) -> Image.Image:
    """
    Load images from local paths and arrange them in a sprite grid.

    Args:
        paths: List of local file paths
        images_per_row: Number of images per row (default: auto-calculate for ~square grid)
        thumbnail_size: Size to resize each image to (width, height)
        background_color: RGB background color
        padding: Pixels between images

    Returns:
        PIL Image containing the sprite grid

    Example:
        sprite = create_sprite_from_paths(
            ["/path/to/img1.jpg", "/path/to/img2.jpg"],
            images_per_row=2,
            thumbnail_size=(128, 128),
        )
        sprite.save("outfit_sprite.png")
    """
    images = []
    for path in paths:
        try:
            img = Image.open(path).convert('RGB')
            img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            images.append(img)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            placeholder = Image.new('RGB', thumbnail_size, (200, 200, 200))
            images.append(placeholder)

    if not images:
        raise ValueError("No images were loaded")

    n_images = len(images)
    if images_per_row is None:
        images_per_row = int(n_images ** 0.5)
        if images_per_row * images_per_row < n_images:
            images_per_row += 1

    n_rows = (n_images + images_per_row - 1) // images_per_row

    cell_width = thumbnail_size[0] + padding
    cell_height = thumbnail_size[1] + padding
    sprite_width = cell_width * images_per_row + padding
    sprite_height = cell_height * n_rows + padding

    sprite = Image.new('RGB', (sprite_width, sprite_height), background_color)

    for idx, img in enumerate(images):
        row = idx // images_per_row
        col = idx % images_per_row

        x = padding + col * cell_width
        y = padding + row * cell_height

        x_offset = (thumbnail_size[0] - img.width) // 2
        y_offset = (thumbnail_size[1] - img.height) // 2

        sprite.paste(img, (x + x_offset, y + y_offset))

    return sprite
