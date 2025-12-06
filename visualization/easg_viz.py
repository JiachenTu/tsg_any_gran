"""
EASG visualization functions with F(t) keyframes and G(t) scene graphs.
"""

from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt

from .data_loader import get_graph_frames
from .graph_viz import draw_scene_graph
from .keyframe_viz import draw_bounding_boxes, display_keyframes
from .utils import format_graph_as_text, extract_verb_dobj
from .config import EASG_FRAMES_DIR, BBOX_FONT_SIZE, BBOX_LINE_WIDTH


def browse_easg_clip(
    clip_id: str,
    easg_data: Dict[str, Any],
    graph_index: int = 0,
    show_all_graphs: bool = False
) -> Optional[Dict[str, Any]]:
    """Browse EASG keyframes for a specific clip.

    Args:
        clip_id: EASG clip ID
        easg_data: Loaded EASG annotations
        graph_index: Which graph to visualize (0-based index)
        show_all_graphs: If True, list all graphs for this clip

    Returns:
        Graph data dict or None if not found
    """
    if clip_id not in easg_data:
        print(f"Error: Clip ID '{clip_id}' not found in EASG dataset.")
        print(f"\nAvailable clip IDs (first 10):")
        for cid in list(easg_data.keys())[:10]:
            print(f"  {cid}")
        return None

    clip_data = easg_data[clip_id]
    graphs = clip_data.get('graphs', [])

    # Header
    print("\n" + "=" * 80)
    print(" EASG CLIP BROWSER")
    print("=" * 80)
    print(f"\nClip ID: {clip_id}")
    print(f"Total graphs in clip: {len(graphs)}")
    print(f"Video UID: {clip_data.get('video_uid', 'N/A')}")
    print(f"Split: {clip_data.get('split', 'N/A')}")

    # List all graphs if requested
    if show_all_graphs:
        print("\n" + "-" * 40)
        print(" ALL GRAPHS IN THIS CLIP")
        print("-" * 40)
        for i, graph in enumerate(graphs):
            triplets = graph.get('triplets', [])
            verb, dobj = extract_verb_dobj(triplets)
            action_str = f"{verb} → {dobj}" if verb and dobj else (verb or "unknown")
            marker = " ← CURRENT" if i == graph_index else ""
            print(f"[{i:2d}] {action_str}{marker}")
        print()

    # Validate graph index
    if graph_index >= len(graphs):
        print(f"Error: Graph index {graph_index} out of range (max: {len(graphs) - 1})")
        return None

    graph = graphs[graph_index]

    # Display selected graph info
    print("\n" + "-" * 40)
    print(f" GRAPH #{graph_index} DETAILS")
    print("-" * 40)
    print(f"Graph UID: {graph.get('graph_uid', 'N/A')}")
    print(f"Timestamps: PRE={graph.get('pre')}, PNR={graph.get('pnr')}, POST={graph.get('post')}")

    print("\nTriplets:")
    for triplet in graph.get('triplets', []):
        if len(triplet) == 3:
            print(f"  {triplet[0]} --[{triplet[1]}]--> {triplet[2]}")

    print("\nGroundings (bounding boxes):")
    groundings = graph.get('groundings', {})
    for frame_type, objects in groundings.items():
        print(f"  {frame_type.upper()}:")
        for obj_name, bbox in objects.items():
            print(f"    {obj_name}: left={bbox['left']}, top={bbox['top']}, "
                  f"width={bbox['width']}, height={bbox['height']}")

    # Display keyframes
    print("\n" + "-" * 40)
    print(" KEYFRAMES F(t)")
    print("-" * 40)
    display_keyframes(clip_id, graph_index, easg_data, show_triplets=False)

    # Draw scene graph
    print("\n" + "-" * 40)
    print(" SCENE GRAPH G(t)")
    print("-" * 40)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    draw_scene_graph(
        graph.get('triplets', []),
        title=f"EASG Graph #{graph_index}: {format_graph_as_text(graph.get('triplets', []))}",
        ax=ax
    )
    plt.tight_layout()
    plt.show()

    return graph


def visualize_easg_graph(
    clip_id: str,
    graph_index: int,
    easg_data: Dict[str, Any],
    frames_dir: str = EASG_FRAMES_DIR,
    font_size: int = BBOX_FONT_SIZE,
    line_width: int = BBOX_LINE_WIDTH
) -> Optional[plt.Figure]:
    """Visualize EASG graph with F(t) keyframes alongside G(t) scene graph.

    This creates a 3x2 layout:
    - Row 0: PRE frame (left) | G(t) scene graph (right)
    - Row 1: PNR frame (left) | G(t) scene graph (right)
    - Row 2: POST frame (left) | G(t) scene graph (right)

    Args:
        clip_id: EASG clip ID
        graph_index: Which graph to visualize (0-based index)
        easg_data: Loaded EASG annotations
        frames_dir: Base directory for frames
        font_size: Font size for bounding box labels
        line_width: Line width for bounding boxes

    Returns:
        Matplotlib figure or None if not found
    """
    frame_info = get_graph_frames(clip_id, graph_index, easg_data, frames_dir)

    if frame_info is None:
        print(f"Could not find frames for clip {clip_id}, graph {graph_index}")
        return None

    triplets = frame_info.get('triplets', [])
    groundings = frame_info.get('groundings', {})
    timestamps = frame_info.get('timestamps', {})

    # Create 3x2 figure
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))

    frame_types = ['pre', 'pnr', 'post']
    frame_titles = [
        ('PRE Frame F(t_pre)', 'Scene Graph G(t)'),
        ('PNR Frame F(t_pnr)', 'Scene Graph G(t)'),
        ('POST Frame F(t_post)', 'Scene Graph G(t)')
    ]

    for row, (frame_type, titles) in enumerate(zip(frame_types, frame_titles)):
        # Left: Keyframe with bounding boxes
        ax_frame = axes[row, 0]
        img = draw_bounding_boxes(
            frame_info[frame_type],
            groundings,
            frame_type,
            font_size=font_size,
            line_width=line_width
        )

        if img is not None:
            ax_frame.imshow(img)
            ts = timestamps.get(frame_type, 'N/A')
            ax_frame.set_title(f"{titles[0]}\nTimestamp: {ts}", fontsize=14, fontweight='bold')
        else:
            ax_frame.text(0.5, 0.5, 'Image not found',
                          ha='center', va='center', transform=ax_frame.transAxes)
            ax_frame.set_title(titles[0], fontsize=14)
        ax_frame.axis('off')

        # Right: Scene graph G(t)
        ax_graph = axes[row, 1]
        draw_scene_graph(
            triplets,
            title=f"{titles[1]}\n{format_graph_as_text(triplets)}",
            ax=ax_graph,
            show_legend=(row == 0)  # Only show legend on first row
        )

    # Add overall title
    verb, dobj = extract_verb_dobj(triplets)
    action = f"{verb} → {dobj}" if verb and dobj else "Unknown action"
    fig.suptitle(
        f"EASG Visualization: {clip_id}\n"
        f"Graph #{graph_index}: {action}",
        fontsize=16, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return fig


def list_easg_clips(
    easg_data: Dict[str, Any],
    limit: Optional[int] = None
) -> None:
    """List all available EASG clip IDs.

    Args:
        easg_data: Loaded EASG annotations
        limit: Optional limit on number of clips to show
    """
    print("\n" + "=" * 60)
    print(" AVAILABLE EASG CLIPS")
    print("=" * 60)
    print(f"Total clips: {len(easg_data)}\n")

    clips = list(easg_data.items())
    clips_to_show = clips[:limit] if limit else clips

    for i, (clip_id, data) in enumerate(clips_to_show):
        n_graphs = len(data.get('graphs', []))
        split = data.get('split', 'N/A')
        print(f"{i:3d}: {clip_id} ({n_graphs} graphs, {split})")

    if limit and len(clips) > limit:
        print(f"\n... and {len(clips) - limit} more clips")


def visualize_easg_sequence(
    clip_id: str,
    easg_data: Dict[str, Any],
    start_index: int = 0,
    num_graphs: int = 5,
    frames_dir: str = EASG_FRAMES_DIR
) -> Optional[plt.Figure]:
    """Visualize a sequence of EASG graphs from a clip.

    Shows PNR frames for multiple consecutive graphs to understand
    the action sequence.

    Args:
        clip_id: EASG clip ID
        easg_data: Loaded EASG annotations
        start_index: Starting graph index
        num_graphs: Number of graphs to show
        frames_dir: Base directory for frames

    Returns:
        Matplotlib figure or None if not found
    """
    if clip_id not in easg_data:
        print(f"Error: Clip ID '{clip_id}' not found")
        return None

    clip_data = easg_data[clip_id]
    graphs = clip_data.get('graphs', [])

    end_index = min(start_index + num_graphs, len(graphs))
    actual_num = end_index - start_index

    if actual_num <= 0:
        print(f"No graphs in range [{start_index}, {end_index})")
        return None

    # Create figure: 2 rows x num_graphs columns
    # Row 0: PNR frames
    # Row 1: Scene graphs
    fig, axes = plt.subplots(2, actual_num, figsize=(5 * actual_num, 10))

    if actual_num == 1:
        axes = axes.reshape(2, 1)

    for col, graph_idx in enumerate(range(start_index, end_index)):
        frame_info = get_graph_frames(clip_id, graph_idx, easg_data, frames_dir)

        if frame_info is None:
            continue

        triplets = frame_info.get('triplets', [])
        verb, dobj = extract_verb_dobj(triplets)
        action = f"{verb} → {dobj}" if verb and dobj else "?"

        # Top row: PNR frame
        ax_frame = axes[0, col]
        img = draw_bounding_boxes(
            frame_info['pnr'],
            frame_info.get('groundings', {}),
            'pnr',
            font_size=20,
            line_width=3
        )
        if img is not None:
            ax_frame.imshow(img)
        ax_frame.set_title(f"G{graph_idx}: {action}", fontsize=10, fontweight='bold')
        ax_frame.axis('off')

        # Bottom row: Scene graph
        ax_graph = axes[1, col]
        draw_scene_graph(
            triplets,
            ax=ax_graph,
            node_size=1200,
            font_size=8,
            edge_font_size=7,
            show_legend=False
        )

    fig.suptitle(
        f"EASG Sequence: {clip_id}\n"
        f"Graphs {start_index} to {end_index - 1}",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return fig
