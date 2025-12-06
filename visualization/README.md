# TSG-Bench & EASG Visualization

Visualization toolkit for TSG-Bench SGQA samples and EASG keyframe annotations.

## Quick Start

```python
import sys
sys.path.insert(0, '/home/jtu9/sgg/tsg-bench')

from visualization import (
    load_sgqa_samples, load_easg_annotations,
    visualize_sgqa_sample, visualize_easg_graph
)

# Load data
sgqa_samples = load_sgqa_samples()
easg_data = load_easg_annotations()

# Visualize SGQA sample (with context graph grid + keyframes)
visualize_sgqa_sample(0, sgqa_samples, easg_data)

# Visualize EASG graph (F(t) keyframes + G(t) scene graph side-by-side)
visualize_easg_graph('e9be1118-a5cf-4431-b2e8-e3edcfa9f949', 5, easg_data)
```

## API Reference

### Data Loading

| Function | Description |
|----------|-------------|
| `load_sgqa_samples()` | Load SGQA samples from jsonl |
| `load_easg_annotations()` | Load EASG master annotations |
| `get_graph_frames(clip_id, graph_index, easg_data)` | Get keyframe paths for a graph |

### SGQA Visualization

| Function | Description |
|----------|-------------|
| `visualize_sgqa_sample(idx, sgqa_samples, easg_data)` | Full sample: text + graph grid + keyframes + Q&A |
| `visualize_random_sgqa_sample(sgqa_samples, easg_data)` | Random sample visualization |
| `display_context_graphs_visual(context_graphs, cols=4)` | NetworkX grid of all context graphs |
| `display_context_graphs_text(context_graphs)` | Text-based graph display |
| `list_sgqa_samples(sgqa_samples, easg_data, limit=10)` | List available samples |

### EASG Visualization

| Function | Description |
|----------|-------------|
| `visualize_easg_graph(clip_id, graph_index, easg_data)` | **3x2 layout: F(t) + G(t) for PRE/PNR/POST** |
| `browse_easg_clip(clip_id, easg_data, graph_index, show_all_graphs)` | Browse clip with graph listing |
| `visualize_easg_sequence(clip_id, easg_data, start_index, num_graphs)` | Sequence of consecutive actions |
| `list_easg_clips(easg_data, limit=10)` | List available clips |

### Low-Level Functions

| Function | Description |
|----------|-------------|
| `draw_scene_graph(triplets, title, ax)` | Draw single NetworkX graph |
| `draw_context_graphs(context_graphs, cols)` | Draw graph grid |
| `draw_bounding_boxes(image_path, groundings, frame_type)` | Draw bboxes on image |
| `display_keyframes(clip_id, graph_index, easg_data)` | Show PRE/PNR/POST frames |

## Examples

### 1. SGQA with Context Graph Grid
```python
# Shows all context graphs as NetworkX visualizations
visualize_sgqa_sample(
    sample_index=4,
    sgqa_samples=sgqa_samples,
    easg_data=easg_data,
    show_text=True,       # Text representation
    show_visual=True,     # NetworkX graph grid
    show_keyframes=True,  # EASG keyframes
    show_qa=True          # Q&A pairs
)
```

### 2. EASG F(t) + G(t) Side-by-Side
```python
# Each row: keyframe (left) | scene graph (right)
# Rows: PRE, PNR, POST
visualize_easg_graph(
    clip_id='e9be1118-a5cf-4431-b2e8-e3edcfa9f949',
    graph_index=10,
    easg_data=easg_data
)
```

### 3. Action Sequence
```python
# Shows PNR frames + graphs for consecutive actions
visualize_easg_sequence(
    clip_id='e9be1118-a5cf-4431-b2e8-e3edcfa9f949',
    easg_data=easg_data,
    start_index=0,
    num_graphs=5
)
```

### 4. Random Exploration
```python
sample, idx = visualize_random_sgqa_sample(sgqa_samples, easg_data)
print(f"Viewed sample {idx}")
```

### 5. Custom Bounding Boxes
```python
from visualization import get_graph_frames, draw_bounding_boxes

frame_info = get_graph_frames(clip_id, 5, easg_data)
img = draw_bounding_boxes(
    frame_info['pnr'],
    frame_info['groundings'],
    'pnr',
    font_size=32,   # Large text
    line_width=6    # Thick lines
)
```

## Package Structure

```
visualization/
├── __init__.py      # Package exports
├── config.py        # Paths, colors, settings
├── data_loader.py   # Data loading
├── utils.py         # Utilities
├── graph_viz.py     # NetworkX visualization
├── keyframe_viz.py  # Keyframe + bbox display
├── sgqa_viz.py      # SGQA functions
├── easg_viz.py      # EASG functions
└── demo.ipynb       # Demo notebook
```

## Configuration

Edit `config.py` to change:
- `SGQA_PATH` - Path to sgqa.jsonl
- `EASG_ANNOT_PATH` - Path to EASG annotations
- `EASG_FRAMES_DIR` - Path to keyframe images
- `BBOX_FONT_SIZE` - Bounding box label size (default: 28)
- `BBOX_LINE_WIDTH` - Bounding box line width (default: 5)
