"""
SGQA sample visualization functions.
"""

import random
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt

from .utils import format_graph_as_text
from .graph_viz import draw_scene_graph, draw_context_graphs
from .keyframe_viz import display_keyframes
from .config import CONTEXT_GRAPH_COLS


def display_context_graphs_text(context_graphs: List[List[List[str]]]) -> None:
    """Display all context graphs as formatted text.

    Args:
        context_graphs: List of graphs, each graph is a list of triplets
    """
    print("\n" + "=" * 60)
    print(f"Context Graphs ({len(context_graphs)} actions):")
    print("=" * 60)
    for i, graph in enumerate(context_graphs, 1):
        print(format_graph_as_text(graph, i))
    print()


def display_context_graphs_visual(
    context_graphs: List[List[List[str]]],
    cols: int = CONTEXT_GRAPH_COLS,
    figsize: Optional[Tuple[int, int]] = None
) -> Optional[plt.Figure]:
    """Display all context graphs as a grid of NetworkX visualizations.

    Args:
        context_graphs: List of graphs, each graph is a list of triplets
        cols: Number of columns in the grid
        figsize: Optional figure size

    Returns:
        Matplotlib figure or None if no graphs
    """
    if not context_graphs:
        print("No context graphs to display")
        return None

    fig = draw_context_graphs(context_graphs, cols=cols, figsize=figsize)
    plt.show()
    return fig


def display_qa_pairs(qa_pairs: List[Dict[str, str]]) -> None:
    """Display Q&A pairs in a formatted way.

    Args:
        qa_pairs: List of {'Q': question, 'A': answer} dictionaries
    """
    print("\n" + "=" * 60)
    print("Q&A Pairs:")
    print("=" * 60)
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\nQ{i}: {qa.get('Q', 'N/A')}")
        print(f"A{i}: {qa.get('A', 'N/A')}")
    print()


def visualize_sgqa_sample(
    sample_index: int,
    sgqa_samples: List[Dict[str, Any]],
    easg_data: Dict[str, Any],
    show_text: bool = True,
    show_visual: bool = True,
    show_keyframes: bool = True,
    show_qa: bool = True,
    keyframe_graph_index: int = 0,
    context_graph_cols: int = CONTEXT_GRAPH_COLS
) -> Optional[Dict[str, Any]]:
    """Visualize a TSG-Bench SGQA sample.

    Args:
        sample_index: Index in SGQA dataset
        sgqa_samples: Loaded SGQA samples list
        easg_data: Loaded EASG annotations
        show_text: Show text-based graph representation
        show_visual: Show NetworkX graph visualization (grid of all context graphs)
        show_keyframes: Display keyframes from EASG
        show_qa: Show Q&A pairs
        keyframe_graph_index: Which EASG graph's keyframes to show
        context_graph_cols: Number of columns for context graph grid

    Returns:
        The sample dictionary or None if index out of range
    """
    if sample_index >= len(sgqa_samples):
        print(f"Error: Sample index {sample_index} out of range (max: {len(sgqa_samples) - 1})")
        return None

    sample = sgqa_samples[sample_index]
    data_id = sample.get('data_id', 'N/A')
    context_graphs = sample.get('context_graphs', [])
    qa_pairs = sample.get('qa_pairs', [])

    # Header
    print("\n" + "#" * 70)
    print(f"# TSG-Bench SGQA Sample #{sample_index}")
    print(f"# Data ID (EASG clip_id): {data_id}")
    print(f"# Number of context graphs: {len(context_graphs)}")
    print(f"# Number of Q&A pairs: {len(qa_pairs)}")
    print("#" * 70)

    # Text-based graph visualization
    if show_text:
        display_context_graphs_text(context_graphs)

    # Visual graph grid (NetworkX visualization of all context graphs)
    if show_visual and context_graphs:
        print("\n" + "=" * 60)
        print("Scene Graph Visualizations (All Context Graphs):")
        print("=" * 60)
        display_context_graphs_visual(context_graphs, cols=context_graph_cols)

    # Display keyframes from EASG
    if show_keyframes:
        if data_id in easg_data:
            print("\n" + "=" * 60)
            print(f"EASG Keyframes (Graph #{keyframe_graph_index}):")
            print("=" * 60)
            print("Note: These are EASG annotations, which differ from TSG-Bench graphs.")
            display_keyframes(data_id, keyframe_graph_index, easg_data)
        else:
            print(f"\nNote: Keyframes not available for data_id: {data_id}")

    # Q&A pairs
    if show_qa:
        display_qa_pairs(qa_pairs)

    return sample


def visualize_random_sgqa_sample(
    sgqa_samples: List[Dict[str, Any]],
    easg_data: Dict[str, Any],
    **kwargs
) -> Tuple[Optional[Dict[str, Any]], int]:
    """Visualize a randomly selected TSG-Bench SGQA sample.

    Args:
        sgqa_samples: Loaded SGQA samples list
        easg_data: Loaded EASG annotations
        **kwargs: Additional arguments passed to visualize_sgqa_sample

    Returns:
        Tuple of (sample dict, sample index)
    """
    if not sgqa_samples:
        print("Error: No SGQA samples loaded")
        return None, -1

    idx = random.randint(0, len(sgqa_samples) - 1)
    print(f"\n{'=' * 70}")
    print(f"Randomly selected sample index: {idx}")
    print(f"{'=' * 70}")

    sample = visualize_sgqa_sample(idx, sgqa_samples, easg_data, **kwargs)
    return sample, idx


def list_sgqa_samples(
    sgqa_samples: List[Dict[str, Any]],
    easg_data: Dict[str, Any],
    limit: Optional[int] = None
) -> None:
    """List all available SGQA samples with their data IDs.

    Args:
        sgqa_samples: Loaded SGQA samples list
        easg_data: Loaded EASG annotations (to check keyframe availability)
        limit: Optional limit on number of samples to show
    """
    print("\n" + "=" * 70)
    print("Available SGQA Samples:")
    print("=" * 70)

    samples_to_show = sgqa_samples[:limit] if limit else sgqa_samples

    for i, sample in enumerate(samples_to_show):
        data_id = sample.get('data_id', 'N/A')
        n_graphs = len(sample.get('context_graphs', []))
        n_qa = len(sample.get('qa_pairs', []))
        has_easg = "[EASG]" if data_id in easg_data else "[----]"
        print(f"{i:3d}: {has_easg} {data_id} ({n_graphs} graphs, {n_qa} QA)")

    if limit and len(sgqa_samples) > limit:
        print(f"\n... and {len(sgqa_samples) - limit} more samples")
