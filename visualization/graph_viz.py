"""
NetworkX scene graph visualization functions.
"""

from typing import List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

from .config import NODE_COLORS, GRAPH_NODE_SIZE, GRAPH_FONT_SIZE, GRAPH_EDGE_FONT_SIZE, CONTEXT_GRAPH_COLS
from .utils import get_node_type, format_graph_as_text


def create_scene_graph(graph_triplets: List[List[str]]) -> nx.DiGraph:
    """Create a NetworkX DiGraph from triplets.

    Args:
        graph_triplets: List of [subject, relation, object] triplets

    Returns:
        NetworkX directed graph with nodes and edges
    """
    G = nx.DiGraph()

    # Find the verb first for proper node type assignment
    verb = None
    for triplet in graph_triplets:
        if len(triplet) == 3 and triplet[1] == 'verb':
            verb = triplet[2]
            break

    # Add nodes and edges
    for triplet in graph_triplets:
        if len(triplet) == 3:
            subj, rel, obj = triplet

            # Add nodes with type information
            if not G.has_node(subj):
                G.add_node(subj, node_type=get_node_type(subj, verb))
            if not G.has_node(obj):
                G.add_node(obj, node_type=get_node_type(obj, verb))

            # Add edge
            G.add_edge(subj, obj, relation=rel)

    return G


def draw_scene_graph(
    graph_triplets: List[List[str]],
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    node_size: int = GRAPH_NODE_SIZE,
    font_size: int = GRAPH_FONT_SIZE,
    edge_font_size: int = GRAPH_EDGE_FONT_SIZE,
    show_legend: bool = True
) -> plt.Axes:
    """Draw a scene graph using NetworkX and matplotlib.

    Args:
        graph_triplets: List of triplets defining the graph
        title: Optional title for the graph
        ax: Matplotlib axes to draw on (creates new if None)
        node_size: Size of nodes
        font_size: Font size for node labels
        edge_font_size: Font size for edge labels
        show_legend: Whether to show the node type legend

    Returns:
        The matplotlib axes object
    """
    G = create_scene_graph(graph_triplets)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, 'Empty graph', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=11, fontweight='bold')
        return ax

    # Get node colors based on type
    node_colors = [
        NODE_COLORS.get(G.nodes[n].get('node_type', 'object'), '#CCCCCC')
        for n in G.nodes()
    ]

    # Layout - use spring layout with fixed seed for reproducibility
    try:
        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    except Exception:
        pos = nx.circular_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_size,
        ax=ax,
        edgecolors='white',
        linewidths=2
    )

    # Draw node labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=font_size,
        font_weight='bold',
        ax=ax
    )

    # Draw edges with arrows
    nx.draw_networkx_edges(
        G, pos,
        edge_color='#555555',
        arrows=True,
        arrowsize=20,
        connectionstyle='arc3,rad=0.1',
        ax=ax,
        width=2
    )

    # Edge labels
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels,
        font_size=edge_font_size,
        ax=ax,
        font_color='#333333'
    )

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    ax.axis('off')

    # Legend
    if show_legend:
        legend_elements = [
            patches.Patch(facecolor=NODE_COLORS['person'], label='Person', edgecolor='white'),
            patches.Patch(facecolor=NODE_COLORS['verb'], label='Verb', edgecolor='white'),
            patches.Patch(facecolor=NODE_COLORS['object'], label='Object', edgecolor='white'),
            patches.Patch(facecolor=NODE_COLORS['hand'], label='Hand', edgecolor='white'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)

    return ax


def draw_context_graphs(
    context_graphs: List[List[List[str]]],
    cols: int = CONTEXT_GRAPH_COLS,
    figsize: Optional[tuple] = None,
    show_legend: bool = False
) -> plt.Figure:
    """Draw all context graphs in a grid layout.

    Args:
        context_graphs: List of graphs, each graph is a list of triplets
        cols: Number of columns in the grid
        figsize: Figure size (auto-calculated if None)
        show_legend: Whether to show legend on each subplot

    Returns:
        The matplotlib figure object
    """
    n_graphs = len(context_graphs)
    if n_graphs == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, 'No context graphs', ha='center', va='center')
        ax.axis('off')
        return fig

    rows = (n_graphs + cols - 1) // cols

    # Calculate figure size based on number of graphs
    if figsize is None:
        figsize = (cols * 4, rows * 3.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle single row/column case
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i, graph in enumerate(context_graphs):
        row = i // cols
        col = i % cols
        ax = axes[row][col]

        # Create short title
        title = f"G{i + 1}: {format_graph_as_text(graph)}"
        if len(title) > 40:
            title = title[:37] + "..."

        draw_scene_graph(
            graph,
            title=title,
            ax=ax,
            node_size=1500,
            font_size=8,
            edge_font_size=7,
            show_legend=show_legend and i == 0  # Only show legend on first graph
        )

    # Hide empty subplots
    for i in range(n_graphs, rows * cols):
        row = i // cols
        col = i % cols
        axes[row][col].axis('off')

    plt.tight_layout()
    return fig
