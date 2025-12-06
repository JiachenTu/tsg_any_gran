"""
Configuration constants for TSG-Bench & EASG visualization.
"""

# =============================================================================
# Data Paths
# =============================================================================

SGQA_PATH = '/home/jtu9/sgg/tsg-bench/resource/dataset/understanding/sgqa.jsonl'
EASG_ANNOT_PATH = '/home/jtu9/sgg/ego4D_download/EASG/dataset_EASG/EASG/EASG_unict_master_final.json'
EASG_FRAMES_DIR = '/home/jtu9/sgg/ego4D_download/EASG/dataset_EASG/EASG/frames'

# =============================================================================
# Color Schemes
# =============================================================================

# Colors for scene graph node types
NODE_COLORS = {
    'person': '#FF6B35',   # Orange
    'verb': '#4ECDC4',     # Teal/Blue
    'object': '#95E88A',   # Green
    'hand': '#FFE66D',     # Yellow
}

# Colors for bounding boxes (cycle through for multiple objects)
BBOX_COLORS = [
    '#FF0000',  # Red
    '#00FF00',  # Green
    '#0000FF',  # Blue
    '#FFFF00',  # Yellow
    '#FF00FF',  # Magenta
    '#00FFFF',  # Cyan
    '#FFA500',  # Orange
    '#800080',  # Purple
    '#00FF7F',  # Spring Green
    '#FF1493',  # Deep Pink
]

# =============================================================================
# Visualization Settings
# =============================================================================

# Bounding box settings (improved for clarity)
BBOX_LINE_WIDTH = 5          # Thicker lines for visibility
BBOX_FONT_SIZE = 28          # Larger font for labels
BBOX_LABEL_PADDING = 8       # Padding around label text

# Scene graph settings
GRAPH_NODE_SIZE = 2500       # Size of nodes in NetworkX graph
GRAPH_FONT_SIZE = 10         # Font size for node labels
GRAPH_EDGE_FONT_SIZE = 9     # Font size for edge labels

# Figure sizes
KEYFRAME_FIGSIZE = (20, 7)   # For PRE/PNR/POST display
GRAPH_FIGSIZE = (12, 8)      # For single scene graph
CONTEXT_GRAPH_COLS = 4       # Columns in context graph grid

# Font paths (fallbacks)
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]
