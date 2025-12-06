"""
TSG-Bench & EASG Visualization Package

A modular visualization toolkit for exploring TSG-Bench SGQA samples
and EASG keyframe annotations with scene graphs.

Usage:
    from visualization import load_sgqa_samples, load_easg_annotations
    from visualization import visualize_sgqa_sample, visualize_easg_graph

    # Load data
    sgqa_samples = load_sgqa_samples()
    easg_data = load_easg_annotations()

    # Visualize SGQA sample with context graphs
    visualize_sgqa_sample(0, sgqa_samples, easg_data)

    # Visualize EASG graph with F(t) + G(t)
    visualize_easg_graph('clip_id', 0, easg_data)
"""

from .config import (
    SGQA_PATH,
    EASG_ANNOT_PATH,
    EASG_FRAMES_DIR,
    NODE_COLORS,
    BBOX_COLORS,
)

from .data_loader import (
    load_sgqa_samples,
    load_easg_annotations,
    get_graph_frames,
)

from .utils import (
    format_triplet,
    format_graph_as_text,
    get_node_type,
)

from .graph_viz import (
    create_scene_graph,
    draw_scene_graph,
    draw_context_graphs,
)

from .keyframe_viz import (
    draw_bounding_boxes,
    display_keyframes,
)

from .sgqa_viz import (
    visualize_sgqa_sample,
    visualize_random_sgqa_sample,
    display_context_graphs_text,
    display_context_graphs_visual,
    display_qa_pairs,
    list_sgqa_samples,
)

from .easg_viz import (
    browse_easg_clip,
    visualize_easg_graph,
    list_easg_clips,
    visualize_easg_sequence,
)

__version__ = "1.0.0"
__all__ = [
    # Config
    "SGQA_PATH",
    "EASG_ANNOT_PATH",
    "EASG_FRAMES_DIR",
    "NODE_COLORS",
    "BBOX_COLORS",
    # Data loading
    "load_sgqa_samples",
    "load_easg_annotations",
    "get_graph_frames",
    # Utils
    "format_triplet",
    "format_graph_as_text",
    "get_node_type",
    # Graph visualization
    "create_scene_graph",
    "draw_scene_graph",
    "draw_context_graphs",
    # Keyframe visualization
    "draw_bounding_boxes",
    "display_keyframes",
    # SGQA visualization
    "visualize_sgqa_sample",
    "visualize_random_sgqa_sample",
    "display_context_graphs_text",
    "display_context_graphs_visual",
    "display_qa_pairs",
    "list_sgqa_samples",
    # EASG visualization
    "browse_easg_clip",
    "visualize_easg_graph",
    "list_easg_clips",
    "visualize_easg_sequence",
]
