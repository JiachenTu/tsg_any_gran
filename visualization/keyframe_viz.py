"""
Keyframe visualization with improved bounding box annotations.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from .config import (
    BBOX_COLORS,
    BBOX_LINE_WIDTH,
    BBOX_FONT_SIZE,
    BBOX_LABEL_PADDING,
    KEYFRAME_FIGSIZE,
    FONT_PATHS,
    EASG_FRAMES_DIR
)
from .data_loader import get_graph_frames


def _get_font(size: int = BBOX_FONT_SIZE) -> ImageFont.FreeTypeFont:
    """Get a TrueType font, falling back to default if not found.

    Args:
        size: Font size

    Returns:
        PIL ImageFont object
    """
    for font_path in FONT_PATHS:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            continue

    # Fallback to default font
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_bounding_boxes(
    image_path: Path,
    groundings: Dict[str, Dict[str, Dict[str, int]]],
    frame_type: str = 'pnr',
    font_size: int = BBOX_FONT_SIZE,
    line_width: int = BBOX_LINE_WIDTH,
    label_padding: int = BBOX_LABEL_PADDING
) -> Optional[Image.Image]:
    """Load image and draw bounding boxes with improved visibility.

    Args:
        image_path: Path to the image file
        groundings: Dict with frame_type -> object_name -> bbox
        frame_type: 'pre', 'pnr', or 'post'
        font_size: Size of label text (default: 28)
        line_width: Width of bounding box lines (default: 5)
        label_padding: Padding around label text (default: 8)

    Returns:
        PIL Image with bounding boxes drawn, or None if image not found
    """
    if not Path(image_path).exists():
        return None

    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Get font
    font = _get_font(font_size)
    small_font = _get_font(max(12, font_size - 8))

    # Get groundings for this frame type
    frame_groundings = groundings.get(frame_type, {})

    for i, (obj_name, bbox) in enumerate(frame_groundings.items()):
        color = BBOX_COLORS[i % len(BBOX_COLORS)]

        # bbox format: {left, top, width, height}
        left = bbox['left']
        top = bbox['top']
        right = left + bbox['width']
        bottom = top + bbox['height']

        # Draw thick bounding box with slight shadow for better visibility
        # Shadow (offset)
        shadow_offset = 2
        draw.rectangle(
            [left + shadow_offset, top + shadow_offset,
             right + shadow_offset, bottom + shadow_offset],
            outline='#000000',
            width=line_width
        )
        # Main rectangle
        draw.rectangle(
            [left, top, right, bottom],
            outline=color,
            width=line_width
        )

        # Calculate label position and size
        if font:
            text_bbox = draw.textbbox((0, 0), obj_name, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        else:
            text_width = len(obj_name) * 10
            text_height = 20

        # Label background position (above the box if possible)
        label_x = left
        label_y = top - text_height - label_padding * 2

        # If label would be off-screen, put it inside the box at the top
        if label_y < 0:
            label_y = top + label_padding

        # Draw label background with padding
        bg_left = label_x
        bg_top = label_y
        bg_right = label_x + text_width + label_padding * 2
        bg_bottom = label_y + text_height + label_padding * 2

        # Draw shadow for background
        draw.rectangle(
            [bg_left + 2, bg_top + 2, bg_right + 2, bg_bottom + 2],
            fill='#000000'
        )
        # Draw background
        draw.rectangle(
            [bg_left, bg_top, bg_right, bg_bottom],
            fill=color
        )

        # Draw text with contrasting color
        text_x = label_x + label_padding
        text_y = label_y + label_padding

        # Draw text shadow for extra contrast
        if font:
            draw.text((text_x + 1, text_y + 1), obj_name, fill='#000000', font=font)
            draw.text((text_x, text_y), obj_name, fill='white', font=font)
        else:
            draw.text((text_x + 1, text_y + 1), obj_name, fill='#000000')
            draw.text((text_x, text_y), obj_name, fill='white')

    return img


def display_keyframes(
    clip_id: str,
    graph_index: int,
    easg_data: Dict[str, Any],
    frames_dir: str = EASG_FRAMES_DIR,
    figsize: tuple = KEYFRAME_FIGSIZE,
    font_size: int = BBOX_FONT_SIZE,
    line_width: int = BBOX_LINE_WIDTH,
    show_triplets: bool = True
) -> Optional[plt.Figure]:
    """Display PRE, PNR, POST keyframes with bounding boxes.

    Args:
        clip_id: The video clip ID
        graph_index: Index of the graph within the clip
        easg_data: Loaded EASG annotations
        frames_dir: Base directory for frames
        figsize: Figure size tuple
        font_size: Font size for bounding box labels
        line_width: Line width for bounding boxes
        show_triplets: Whether to print triplets below the figure

    Returns:
        Matplotlib figure or None if frames not found
    """
    frame_info = get_graph_frames(clip_id, graph_index, easg_data, frames_dir)

    if frame_info is None:
        print(f"Could not find frames for clip {clip_id}, graph {graph_index}")
        return None

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    frame_types = ['pre', 'pnr', 'post']
    titles = [
        'PRE (Before Action)',
        'PNR (Point of No Return)',
        'POST (After Action)'
    ]

    for ax, frame_type, title in zip(axes, frame_types, titles):
        img = draw_bounding_boxes(
            frame_info[frame_type],
            frame_info['groundings'],
            frame_type,
            font_size=font_size,
            line_width=line_width
        )

        if img is not None:
            ax.imshow(img)
            timestamp = frame_info['timestamps'].get(frame_type, 'N/A')
            ax.set_title(f"{title}\nFrame: {timestamp}", fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Image not found',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=12)

        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Show EASG triplets for this graph
    if show_triplets:
        print("\nEASG Triplets for this graph:")
        for triplet in frame_info.get('triplets', []):
            if len(triplet) == 3:
                print(f"  {triplet[0]} --[{triplet[1]}]--> {triplet[2]}")
            else:
                print(f"  {triplet}")

    return fig


def display_single_keyframe(
    clip_id: str,
    graph_index: int,
    frame_type: str,
    easg_data: Dict[str, Any],
    frames_dir: str = EASG_FRAMES_DIR,
    figsize: tuple = (10, 8),
    font_size: int = BBOX_FONT_SIZE,
    line_width: int = BBOX_LINE_WIDTH,
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """Display a single keyframe (PRE, PNR, or POST) with bounding boxes.

    Args:
        clip_id: The video clip ID
        graph_index: Index of the graph within the clip
        frame_type: 'pre', 'pnr', or 'post'
        easg_data: Loaded EASG annotations
        frames_dir: Base directory for frames
        figsize: Figure size (used if ax is None)
        font_size: Font size for bounding box labels
        line_width: Line width for bounding boxes
        ax: Optional axes to draw on

    Returns:
        Matplotlib axes or None if frame not found
    """
    frame_info = get_graph_frames(clip_id, graph_index, easg_data, frames_dir)

    if frame_info is None:
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    img = draw_bounding_boxes(
        frame_info[frame_type],
        frame_info['groundings'],
        frame_type,
        font_size=font_size,
        line_width=line_width
    )

    if img is not None:
        ax.imshow(img)
        timestamp = frame_info['timestamps'].get(frame_type, 'N/A')
        title_map = {
            'pre': 'PRE (Before Action)',
            'pnr': 'PNR (Point of No Return)',
            'post': 'POST (After Action)'
        }
        ax.set_title(f"{title_map.get(frame_type, frame_type)}\nFrame: {timestamp}",
                     fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Image not found',
                ha='center', va='center', transform=ax.transAxes)

    ax.axis('off')
    return ax
