"""
Data loading functions for TSG-Bench SGQA and EASG datasets.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import SGQA_PATH, EASG_ANNOT_PATH, EASG_FRAMES_DIR


def load_sgqa_samples(sgqa_path: str = SGQA_PATH) -> List[Dict[str, Any]]:
    """Load all SGQA samples from jsonl file.

    Args:
        sgqa_path: Path to the SGQA jsonl file

    Returns:
        List of SGQA sample dictionaries with keys:
        - data_id: clip ID linking to EASG
        - context_graphs: list of scene graphs (each is list of triplets)
        - qa_pairs: list of {Q, A} dictionaries
    """
    samples = []
    with open(sgqa_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f'Loaded {len(samples)} SGQA samples')
    return samples


def load_easg_annotations(easg_path: str = EASG_ANNOT_PATH) -> Dict[str, Any]:
    """Load EASG master annotations.

    Args:
        easg_path: Path to the EASG master JSON file

    Returns:
        Dictionary mapping clip_id to annotation data:
        - graphs: list of graph annotations
        - split: train/val/test
        - video_uid: original Ego4D video UID
        - summaries: text summaries (if available)
    """
    with open(easg_path, 'r') as f:
        data = json.load(f)
    print(f'Loaded EASG annotations for {len(data)} clips')
    return data


def get_graph_frames(
    clip_id: str,
    graph_index: int,
    easg_data: Dict[str, Any],
    frames_dir: str = EASG_FRAMES_DIR
) -> Optional[Dict[str, Any]]:
    """Get keyframe paths and groundings for a specific graph.

    Args:
        clip_id: The video clip ID (e.g., 'e9be1118-a5cf-4431-b2e8-e3edcfa9f949')
        graph_index: Index of the graph within the clip (0-based)
        easg_data: Loaded EASG annotations dictionary
        frames_dir: Base directory for frame images

    Returns:
        Dictionary with:
        - pre: Path to PRE frame
        - pnr: Path to PNR frame
        - post: Path to POST frame
        - groundings: Dict of frame_type -> object_name -> bbox
        - triplets: List of triplets for this graph
        - timestamps: Dict with pre/pnr/post frame numbers
        - graph_uid: Unique identifier for this graph

        Returns None if clip or graph not found
    """
    if clip_id not in easg_data:
        return None

    graphs = easg_data[clip_id].get('graphs', [])
    if graph_index >= len(graphs):
        return None

    graph = graphs[graph_index]
    graph_uid = graph.get('graph_uid', '')

    frame_dir = Path(frames_dir) / graph_uid
    if not frame_dir.exists():
        return None

    return {
        'pre': frame_dir / f'{clip_id}_pre.jpg',
        'pnr': frame_dir / f'{clip_id}_pnr.jpg',
        'post': frame_dir / f'{clip_id}_post.jpg',
        'groundings': graph.get('groundings', {}),
        'triplets': graph.get('triplets', []),
        'timestamps': {
            'pre': graph.get('pre'),
            'pnr': graph.get('pnr'),
            'post': graph.get('post')
        },
        'graph_uid': graph_uid
    }


def get_clip_info(clip_id: str, easg_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get summary information about a clip.

    Args:
        clip_id: The video clip ID
        easg_data: Loaded EASG annotations dictionary

    Returns:
        Dictionary with clip metadata or None if not found
    """
    if clip_id not in easg_data:
        return None

    clip_data = easg_data[clip_id]
    return {
        'clip_id': clip_id,
        'n_graphs': len(clip_data.get('graphs', [])),
        'split': clip_data.get('split', 'unknown'),
        'video_uid': clip_data.get('video_uid', 'unknown'),
        'summaries': clip_data.get('summaries', []),
    }
