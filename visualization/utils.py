"""
Shared utility functions for TSG-Bench & EASG visualization.
"""

from typing import List, Optional, Tuple, Any


def format_triplet(triplet: List[str]) -> str:
    """Format a single triplet as readable text.

    Args:
        triplet: A triplet [subject, relation, object]
                 e.g., ['pick-up', 'with', 'hand1'] or ['person', 'verb', 'pick-up']

    Returns:
        Formatted string like "subject --[relation]--> object"
    """
    if len(triplet) == 3:
        return f"{triplet[0]} --[{triplet[1]}]--> {triplet[2]}"
    return str(triplet)


def format_graph_as_text(graph_triplets: List[List[str]], index: Optional[int] = None) -> str:
    """Format a scene graph (list of triplets) as readable text.

    Extracts the main action pattern: person → verb → object (modifiers)

    Args:
        graph_triplets: List of triplets for one action/graph
        index: Optional action index for display prefix

    Returns:
        Formatted string like "[1] person → pick-up → cup (with hand1)"
    """
    verb = None
    dobj = None
    modifiers = []

    for triplet in graph_triplets:
        if len(triplet) == 3:
            subj, rel, obj = triplet
            if rel == 'verb':
                verb = obj
            elif rel == 'dobj':
                dobj = obj
            elif subj == verb and rel not in ['verb', 'dobj']:
                modifiers.append(f"{rel} {obj}")

    # Build readable string
    prefix = f"[{index}] " if index is not None else ""

    if verb and dobj:
        mod_str = f" ({', '.join(modifiers)})" if modifiers else ""
        return f"{prefix}person → {verb} → {dobj}{mod_str}"
    elif verb:
        mod_str = f" ({', '.join(modifiers)})" if modifiers else ""
        return f"{prefix}person → {verb}{mod_str}"
    else:
        # Fallback: show all triplets
        return f"{prefix}" + "; ".join([format_triplet(t) for t in graph_triplets])


def get_node_type(node_name: str, verb: Optional[str] = None) -> str:
    """Determine the semantic type of a node for coloring.

    Args:
        node_name: The name of the node
        verb: The verb in the current graph (to identify verb nodes)

    Returns:
        One of: 'person', 'verb', 'hand', 'object'
    """
    node_lower = node_name.lower()

    if node_lower == 'person' or node_lower == 'cw':
        return 'person'
    elif 'hand' in node_lower:
        return 'hand'
    elif node_name == verb:
        return 'verb'
    else:
        return 'object'


def extract_verb_dobj(triplets: List[List[str]]) -> Tuple[Optional[str], Optional[str]]:
    """Extract the main verb and direct object from triplets.

    Args:
        triplets: List of triplets

    Returns:
        Tuple of (verb, dobj) or (None, None) if not found
    """
    verb = None
    dobj = None

    for triplet in triplets:
        if len(triplet) == 3:
            if triplet[1] == 'verb':
                verb = triplet[2]
            elif triplet[1] == 'dobj':
                dobj = triplet[2]

    return verb, dobj


def truncate_text(text: str, max_length: int = 20) -> str:
    """Truncate text with ellipsis if too long.

    Args:
        text: Input text
        max_length: Maximum length before truncation

    Returns:
        Truncated text with '...' if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
