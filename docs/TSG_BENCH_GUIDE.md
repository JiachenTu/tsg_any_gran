# TSG-Bench Technical Guide

A comprehensive technical documentation for the TSG-Bench (Text-Scene Graph Benchmark) repository.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Environment Setup](#2-environment-setup)
3. [Repository Structure](#3-repository-structure)
4. [Dataset Structure](#4-dataset-structure)
5. [Scene Graph Format](#5-scene-graph-format)
6. [Understanding Tasks](#6-understanding-tasks)
7. [Generation Tasks](#7-generation-tasks)
8. [Evaluation Pipeline](#8-evaluation-pipeline)
9. [Key Classes Reference](#9-key-classes-reference)
10. [Supported Models & API Cost Estimation](#10-supported-models--api-cost-estimation)
11. [Running Evaluations](#11-running-evaluations)

---

## 1. Overview

### What is TSG-Bench?

TSG-Bench is an open-source evaluation framework for assessing Large Language Models (LLMs) on their ability to **understand** and **generate** spatio-temporal scene graphs from textual narratives.

### The Four Evaluation Tasks

| Task | Type | Description | Samples | Metric |
|------|------|-------------|---------|--------|
| **SGQA** | Understanding | Scene Graph Question Answering | 100 (500 QA pairs) | Accuracy |
| **SGDS** | Understanding | Scene Graph Description Selection | 250 | Accuracy |
| **SA-SGG** | Generation | Single-Action Scene Graph Generation | 1,188 | P/R/F1 |
| **MA-SGG** | Generation | Multi-Action Scene Graph Generation | 853 | P/R/F1 |

### Dataset Scale

- **18 domains** (repair, cooking, assembly, cleaning, etc.)
- **120 scenarios**
- **2,041 textual descriptions**
- **4,289 scene graphs**
- **~15K nodes, ~12K edges**

### Source Data

Built on the **Ego-centric Action Scene Graphs (EASG)** dataset with human-in-the-loop annotation.

---

## 2. Environment Setup

### Conda Environment

The repository uses a conda environment named `tsg-bench` with Python 3.10.

```bash
# Create the environment
conda create -n tsg-bench python=3.10 -y

# Activate
conda activate tsg-bench

# Install base dependencies
pip install -r requirements.txt

# Install additional LangChain integrations
pip install langchain-openai langchain-anthropic

# Install Jupyter kernel (optional, for notebooks)
pip install ipykernel
python -m ipykernel install --user --name tsg-bench --display-name "Python (tsg-bench)"
```

### Dependencies (requirements.txt)

```
langchain-core>=0.1.0
openai>=1.0.0
tqdm>=4.0.0
numpy>=1.23.0
pandas>=1.5.0
pyyaml>=6.0
matplotlib>=3.5.0
anthropic>=0.21.0
```

### Configuration

API keys are stored in `conf.d/config.yaml`:

```yaml
openai:
  key: <your-openai-api-key>
anthropic:
  key: <your-anthropic-api-key>
openrouter:
  key: <your-openrouter-api-key>
```

Use `conf.d/config.example.yaml` as a template.

---

## 3. Repository Structure

```
tsg-bench/
├── conf.d/                          # Configuration files
│   ├── config.yaml                  # API keys (gitignored)
│   └── config.example.yaml          # Template
├── evaluation/                      # Evaluation scripts
│   ├── generation/
│   │   ├── sa-sgg.py               # Single-Action SGG evaluation
│   │   └── ma-sgg.py               # Multi-Action SGG evaluation
│   └── understanding/
│       ├── sgqa.py                 # Question Answering evaluation
│       └── sgds.py                 # Description Selection evaluation
├── models/
│   └── models.py                   # LLM wrapper classes
├── resource/
│   ├── dataset/
│   │   ├── generation/
│   │   │   ├── sa-sgg.jsonl        # SA-SGG dataset (1,188 samples)
│   │   │   └── ma-sgg.jsonl        # MA-SGG dataset (853 samples)
│   │   └── understanding/
│   │       ├── sgqa.jsonl          # SGQA dataset (100 samples)
│   │       └── sgds.jsonl          # SGDS dataset (250 samples)
│   └── prompts/
│       ├── sa-sgg.txt              # SA-SGG prompt template
│       ├── ma-sgg.txt              # MA-SGG prompt template
│       ├── sgqa.txt                # SGQA prompt template
│       └── sgds.txt                # SGDS prompt template
├── utils/
│   ├── config.py                   # Config loading utilities
│   └── path.py                     # Path and file utilities
├── docs/                           # Documentation
└── requirements.txt
```

---

## 4. Dataset Structure

All datasets use **JSONL format** (one JSON object per line).

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `data_id` | string | Unique scenario identifier (same across related samples) |
| `doc_index` | int | Narrative variant index (different ways to describe same scenario) |
| `text_part_index` | int | Sentence position within the full narrative |
| `context` | string | Previous narrative text for understanding |
| `target_sentence` | string | **di** - The description to process |
| `graphs` | array | **Gi** - Ground truth scene graph(s) |
| `graph_id` | array[int] | Action IDs corresponding to each graph |
| `context_graphs` | array | Previous action graphs for context |
| `mandatory_space` | object | Allowed vocabulary constraints |
| `is_repeatable` | bool | Whether the action can repeat |

### Data Hierarchy

```
data_id (Scenario)
  └── doc_index (Narrative variant)
        └── text_part_index (Sentence position)
              └── target_sentence (di)
                    └── graphs (Gi)
```

### SA-SGG Sample Schema

```json
{
  "data_id": "e9be1118-a5cf-4431-b2e8-e3edcfa9f949",
  "doc_index": 1,
  "text_part_index": 4,
  "context": "The task began by preparing the necessary tools...",
  "target_sentence": "The fastener was then secured tightly with both hands.",
  "graph_id": [5],
  "is_repeatable": false,
  "graphs": [
    {
      "action_id": 5,
      "triplets": [
        ["person", "verb", "tighten"],
        ["tighten", "dobj", "clamp"],
        ["tighten", "with", "hand1"],
        ["tighten", "with", "hand2"]
      ],
      "objects": ["tighten", "hand2", "clamp", "person", "hand1"],
      "relationships": ["verb", "dobj", "with", "with"],
      "object_pairs": [["person", "tighten"], ["tighten", "clamp"], ...]
    }
  ],
  "verbs": ["tighten"],
  "mandatory_space": {
    "verb": ["align", "drill", "hold", "loosen", "pick-up", "place", ...],
    "relationship": ["dobj", "from", "into", "on", "with", "verb"],
    "object": ["person", "clamp", "hand1", "hand2", "screwdriver", ...]
  },
  "context_graphs": [...]
}
```

### MA-SGG Sample Schema

Same as SA-SGG, but `graphs` array contains **multiple** scene graphs:

```json
{
  "target_sentence": "The screw was then positioned and secured into the wood using the screwdriver.",
  "graph_id": [15, 16],
  "graphs": [
    {
      "action_id": 15,
      "triplets": [
        ["person", "verb", "position"],
        ["position", "dobj", "screw"],
        ["position", "on", "wood"],
        ["position", "with", "hand2"]
      ]
    },
    {
      "action_id": 16,
      "triplets": [
        ["person", "verb", "screw"],
        ["screw", "dobj", "screw"],
        ["screw", "into", "wood"],
        ["screw", "with", "screwdriver"],
        ["screw", "with", "hand1"]
      ]
    }
  ]
}
```

### SGQA Sample Schema

```json
{
  "data_id": "19cc4e42-39bb-41f9-b9de-9f2940eed6a2",
  "doc_index": 2,
  "text_part_index": 9,
  "context_graphs": [
    [["pick-up", "with", "hand1"], ["pick-up", "dobj", "mop-stick"], ...],
    [["sweep", "with", "hand1"], ["sweep", "dobj", "floor"], ...]
  ],
  "qa_pairs": [
    {"Q": "What object was picked up before sweeping the floor?", "A": "mop-stick"},
    {"Q": "Which location did the person interact with after using the cloth?", "A": "wall"}
  ]
}
```

### SGDS Sample Schema

```json
{
  "target_sentence": "This step was repeated to achieve the desired consistency.",
  "position": 1,
  "variations": [
    "This step was skipped to avoid achieving the desired consistency.",
    "This step was repeated to achieve the desired consistency.",
    "This step was repeated to disrupt the desired consistency.",
    "This step was altered to prevent the desired consistency.",
    "This step was ignored to ensure the undesired consistency."
  ],
  "triplet": [
    ["person", "verb", "roll"],
    ["roll", "dobj", "dough"],
    ["roll", "with", "roller"],
    ["roll", "with", "hand1"],
    ["roll", "with", "hand2"]
  ],
  "context_graphs": [...]
}
```

---

## 5. Scene Graph Format

### Triplet Structure

A scene graph is represented as a list of **triplets**:

```
[subject, edge, object]
```

Visual representation:
```
subject --> edge --> object
```

### Node Types

| Type | Description | Examples |
|------|-------------|----------|
| `person` | The actor (always root) | `person` |
| `action` | Verb/activity nodes | `pick-up`, `place`, `tighten`, `drill` |
| `object` | Physical entities | `screwdriver`, `wood`, `clamp`, `bowl` |
| `hand` | Body parts for manipulation | `hand1` (left), `hand2` (right) |

### Edge Types

| Edge | Description | Example |
|------|-------------|---------|
| `verb` | Connects person to action | `person -> verb -> pick-up` |
| `dobj` | Direct object of action | `pick-up -> dobj -> screwdriver` |
| `with` | Instrument/tool used | `pick-up -> with -> hand1` |
| `from` | Source location | `pick-up -> from -> table` |
| `on` | Target surface | `place -> on -> workbench` |
| `into` | Target container | `screw -> into -> wood` |
| `to` | Destination | `carry -> to -> board` |

### Hand Tracking Rules

The dataset tracks which hand holds which object:

1. **First grasped object** → `hand1`
2. **Second grasped object** → `hand2`
3. **After release** → reset to `hand1`
4. **Both hands used** → explicitly list both `hand1` AND `hand2`

### Example: Complete Scene Graph

**Description:** "Pick up the screwdriver with your left hand"

**Scene Graph:**
```
person -> verb -> pick-up
pick-up -> dobj -> screwdriver
pick-up -> with -> hand1
```

**As JSON:**
```json
{
  "triplets": [
    ["person", "verb", "pick-up"],
    ["pick-up", "dobj", "screwdriver"],
    ["pick-up", "with", "hand1"]
  ]
}
```

---

## 6. Understanding Tasks

### 6.1 SGQA (Scene Graph Question Answering)

**Task:** Answer questions by reasoning over a sequence of scene graphs.

**File:** `evaluation/understanding/sgqa.py`

#### Input/Output

- **Input:** Sequence of scene graphs + Question
- **Output:** Single word answer in brackets: `[answer]`
- **Metric:** Accuracy (case-insensitive exact match)

#### Prompt Template (`resource/prompts/sgqa.txt`)

```
You are a highly advanced language model specialized in answering questions
based on a given scene graph and question. Your task is to analyze the scene
graph and provide the correct answer in a single word. Your output must strictly
follow the format [answer], and nothing else should be printed.

Scene Graph: {scene_graph}
Question: {question}
```

#### Key Classes

**`QA`** - Inference class
```python
class QA:
    def __init__(self, model: LLM):
        # Loads prompt template
        # Initializes model

    def invoke(self, scene_graph: str, question: str) -> str:
        # Formats prompt with scene_graph and question
        # Calls model.invoke()
        # Extracts answer from brackets using regex: r"\[(.*?)\]"
        # Returns extracted answer or full response
```

**`QADataLoader`** - Data loading
```python
class QADataLoader:
    def __init__(self):
        # Loads sgqa.jsonl

    def get_data(self) -> List[Dict]:
        # Flattens qa_pairs into individual items
        # Returns list where each item has one Q/A pair
```

**`QAEvaluator`** - Evaluation orchestration
```python
class QAEvaluator:
    def __init__(self, model: LLM):
        # Creates QA instance

    def process_single_question(self, data: Dict) -> Dict:
        # Invokes QA for one question
        # Compares prediction with ground truth
        # Returns result dict with is_correct flag

    def evaluate(self, data_loader: QADataLoader) -> Dict:
        # Uses ThreadPoolExecutor (5 workers)
        # Processes all questions in parallel
        # Returns: {model, accuracy, total_correct, total_questions}
```

#### Evaluation Flow

```
sgqa.jsonl
    ↓
QADataLoader.get_data() → Flatten qa_pairs
    ↓
QAEvaluator.evaluate()
    ↓
ThreadPoolExecutor (5 workers)
    ↓
QA.invoke(scene_graph, question)
    ↓
regex extract [answer]
    ↓
Compare with ground truth (case-insensitive)
    ↓
Calculate accuracy = correct / total
```

#### Example

```python
# Input
scene_graph = "[['pick-up', 'dobj', 'mop-stick'], ['sweep', 'dobj', 'floor']]"
question = "What object was picked up before sweeping the floor?"

# Model Output
"[mop-stick]"

# Extracted Answer
"mop-stick"

# Ground Truth
"mop-stick"

# Result
is_correct = True
```

---

### 6.2 SGDS (Scene Graph Description Selection)

**Task:** Select the best description (A-E) for a given scene graph.

**File:** `evaluation/understanding/sgds.py`

#### Input/Output

- **Input:** Target scene graph + 5 candidate descriptions + Context graphs
- **Output:** Letter A-E in brackets: `[A]`, `[B]`, etc.
- **Metric:** Accuracy

#### Prompt Template (`resource/prompts/sgds.txt`)

```
You are an AI that analyzes a Scene Graph based on the context and select the
best text description of it among the given candidates.

1. Input:
   - Context: A list of scene graphs representing the preceding context.
   - Target Scene Graph: A set of triplets to describe.
   - Description Candidates: 5 candidate descriptions.

2. Task: Determine which description best matches the Target Scene Graph.

3. Output: Only the letter in [ ] (e.g., [A]).

Key rules of edges in a triplet:
   - `verb` describes the action performed by `person`.
   - `dobj` links the action to its direct object.
   - Other edges describe spatial relationships.

Input:
- Context: {context}
- Target Scene Graph: {triplet}
- Description Candidates:
{sentences}
```

#### Key Classes

**`SceneGraphToText`** - Inference class
```python
class SceneGraphToText:
    def __init__(self, model: LLM):
        # Loads prompt template

    def invoke(self, sentences: str, triplet: str, context_graphs: str) -> str:
        # Formats prompt
        # Returns model response
```

**`SceneGraphEvaluator`** - Evaluation orchestration
```python
class SceneGraphEvaluator:
    def __init__(self, model: LLM):
        # Creates SceneGraphToText instance

    def parse_prediction(self, response: str) -> Optional[int]:
        # Regex: r"\[([A-E])\]|\b([A-E])\b"
        # Converts letter to index: ord(letter) - 65
        # A=0, B=1, C=2, D=3, E=4

    def process_single_data(self, data: Dict) -> Dict:
        # Formats variations as "A: ...\nB: ...\n..."
        # Invokes model
        # Parses prediction
        # Compares with data["position"]

    def evaluate_dataset(self, input_path: str) -> Dict:
        # ThreadPoolExecutor (15 workers)
        # Returns: {model, accuracy, total_samples, correct_predictions}
```

#### Evaluation Flow

```
sgds.jsonl
    ↓
Load samples
    ↓
SceneGraphEvaluator.evaluate_dataset()
    ↓
ThreadPoolExecutor (15 workers)
    ↓
Format variations as A-E options
    ↓
SceneGraphToText.invoke()
    ↓
parse_prediction() → letter to index
    ↓
Compare with ground truth position
    ↓
Calculate accuracy = correct / total
```

#### Example

```python
# Input
triplet = "[['person', 'verb', 'roll'], ['roll', 'dobj', 'dough']]"
variations = [
    "A: This step was skipped...",
    "B: This step was repeated to achieve...",  # Correct
    "C: This step was repeated to disrupt...",
    "D: This step was altered...",
    "E: This step was ignored..."
]
position = 1  # Ground truth: B

# Model Output
"[B]"

# Parsed
predicted_index = 1

# Result
is_correct = (1 == 1) = True
```

---

## 7. Generation Tasks

### 7.1 SA-SGG (Single-Action Scene Graph Generation)

**Task:** Generate ONE scene graph from a single-action sentence.

**File:** `evaluation/generation/sa-sgg.py`

#### Input/Output

- **Input:** Context, target sentence, vocabulary constraints
- **Output:** Triplets in format `node -> edge -> node` (one per line)
- **Metrics:** Precision, Recall, F1 (macro-averaged)

#### Prompt Template Excerpt (`resource/prompts/sa-sgg.txt`)

```
Rules for Scene Graph Representation:
1. A graph is composed of one or more triplets of nodes and edges.
2. A triplet starts with a node and another node is connected by an edge.
   Format: node -> edge -> node
3. Each triplet is split with a new line.
4. There must be a triplet containing the 'person' node.

Rules for Nodes:
1. All nodes must be sourced from the available_nodes.
2. Use the exact name.

Rules for Edges:
1. All edges must be sourced from the available_edges.
2. The 'verb' edge must connect [person] to [action].
3. The 'dobj' edge connects [action] to [object].

Context: {context}
Target Sentence: {target_sentence}
Available Nodes: {available_nodes}
Available Edges: {available_edges}
Available Verbs: {verbs}
```

#### Key Classes

**`GraphGeneration`** - Inference class
```python
class GraphGeneration:
    def __init__(self, model: LLM):
        # Loads sa-sgg.txt prompt template

    def invoke(self, context, target_sentence, available_nodes,
               available_edges, verbs) -> str:
        # Formats prompt
        # Returns model.invoke(prompt)
```

**`GraphScorer`** - Static scoring methods
```python
class GraphScorer:
    @staticmethod
    def parse_response(response: str) -> List[List[List[str]]]:
        """
        Parse model response into scene graphs.

        Input: "person -> verb -> pick-up\npick-up -> dobj -> screwdriver"
        Output: [[["person", "verb", "pick-up"], ["pick-up", "dobj", "screwdriver"]]]

        Rules:
        - Split by newlines
        - Empty lines separate different graphs
        - Each line split by " -> " into 3 parts
        - Only valid 3-part triplets are kept
        """

    @staticmethod
    def calculate_scores(true_triplets: List, pred_triplets: List) -> Dict:
        """
        Calculate precision, recall, F1 for triplet comparison.

        Algorithm:
        1. Convert triplets to strings: ["a", "b", "c"] -> "a b c"
        2. Create sets for both true and predicted
        3. Find intersection (correct predictions)
        4. Calculate metrics
        """
        true_set = set([" ".join(t) for t in true_triplets])
        pred_set = set([" ".join(t) for t in pred_triplets])

        correct = len(true_set & pred_set)

        precision = correct / len(pred_set) if pred_set else 0
        recall = correct / len(true_set) if true_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {"precision": precision, "recall": recall, "f1": f1}
```

**`GraphEvaluator`** - Evaluation orchestration
```python
class GraphEvaluator:
    def __init__(self, model: LLM):
        # Creates GraphGeneration instance

    def process_single_data(self, data: Dict) -> Dict:
        # Extracts inputs from data
        # Invokes model
        # Parses response
        # Calculates scores per graph
        # Returns detailed result with missing/incorrect triplets

    def evaluate(self, data_path: str) -> Dict:
        # ThreadPoolExecutor (15 workers)
        # Aggregates macro metrics
        # Returns: {model, macro_precision, macro_recall, macro_f1}
```

#### Metrics Formulas

```
TP (True Positives) = |predicted ∩ ground_truth|

Precision = TP / |predicted|
           "Of all predicted triplets, how many were correct?"

Recall = TP / |ground_truth|
        "Of all true triplets, how many were predicted?"

F1 = 2 × (Precision × Recall) / (Precision + Recall)
     "Harmonic mean of precision and recall"

Macro-Precision = mean(precision_i for all graphs)
Macro-Recall = mean(recall_i for all graphs)
Macro-F1 = mean(f1_i for all graphs)
```

#### Example

```python
# Ground Truth
true_triplets = [
    ["person", "verb", "pick-up"],
    ["pick-up", "dobj", "screwdriver"],
    ["pick-up", "with", "hand1"]
]

# Model Prediction
pred_triplets = [
    ["person", "verb", "pick-up"],      # Correct
    ["pick-up", "dobj", "screwdriver"], # Correct
    ["pick-up", "with", "hand2"]        # Wrong (hand2 vs hand1)
]

# Calculation
true_set = {"person verb pick-up", "pick-up dobj screwdriver", "pick-up with hand1"}
pred_set = {"person verb pick-up", "pick-up dobj screwdriver", "pick-up with hand2"}

correct = 2  # intersection

precision = 2/3 = 0.667
recall = 2/3 = 0.667
f1 = 2 * (0.667 * 0.667) / (0.667 + 0.667) = 0.667
```

---

### 7.2 MA-SGG (Multi-Action Scene Graph Generation)

**Task:** Generate MULTIPLE scene graphs from a complex sentence with implicit actions.

**File:** `evaluation/generation/ma-sgg.py`

#### Key Differences from SA-SGG

| Aspect | SA-SGG | MA-SGG |
|--------|--------|--------|
| Graphs per sample | 1 | N (≥2) |
| Output format | Single triplet block | Multiple blocks separated by blank lines |
| Challenge | Accurate representation | Action decomposition + representation |
| Temperature | 0.1 (default) | 1.0 (more diverse) |

#### Prompt Template Additions

MA-SGG prompt includes:

```
Generate precisely {num_scene_graphs} scene graphs—no more, no less.

If the sentence contains fewer visible actions, additional relevant actions
may be inferred from the context or implicit meaning of the sentence.
```

#### Output Format

```
# Graph 1
person -> verb -> pick-up
pick-up -> dobj -> shirt
pick-up -> with -> hand1

# Graph 2 (separated by blank line)
person -> verb -> place
place -> dobj -> shirt
place -> on -> board
```

#### Implicit Action Decomposition

**Input:** "The garment was positioned on the panel with both hands"

**Decomposed Actions:**
1. **pick-up** - Pick up the garment
2. **carry** - Carry to the panel
3. **place** - Place on the panel

**Generated Graphs:**
```json
[
  {"action_id": 1, "triplets": [["person", "verb", "pick-up"], ...]},
  {"action_id": 2, "triplets": [["person", "verb", "carry"], ...]},
  {"action_id": 3, "triplets": [["person", "verb", "place"], ...]}
]
```

---

## 8. Evaluation Pipeline

### Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dataset (JSONL)                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Evaluator.evaluate(data_path)                  │
│  - Loads JSONL file                                         │
│  - Initializes ThreadPoolExecutor                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           ThreadPoolExecutor (5-15 workers)                 │
│  - Parallel processing of samples                           │
│  - Uses futures.as_completed()                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            process_single_data(sample)                      │
│  - Formats prompt with sample data                          │
│  - Calls model.invoke()                                     │
│  - Parses response                                          │
│  - Calculates per-sample metrics                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               Model.invoke(prompt)                          │
│  - Sends request to LLM API                                 │
│  - Returns response text                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Parse & Score Response                         │
│  - Understanding: Extract answer, compare                   │
│  - Generation: Parse triplets, calculate P/R/F1             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Aggregate Results                              │
│  - Understanding: accuracy = correct / total                │
│  - Generation: macro P/R/F1 = mean across samples           │
└─────────────────────────────────────────────────────────────┘
```

### Parallelization

All evaluators use `concurrent.futures.ThreadPoolExecutor`:

| Task | Workers | Reason |
|------|---------|--------|
| SGQA | 5 | More complex reasoning per sample |
| SGDS | 15 | Simpler selection task |
| SA-SGG | 15 | Standard generation |
| MA-SGG | 15 | Standard generation |

---

## 9. Key Classes Reference

### Model Wrappers (`models/models.py`)

**Base Class:**
```python
class LLM:
    def __init__(self):
        self.config = get_config()

    def invoke(self, text: str) -> str:
        raise NotImplementedError
```

**Implementations:**

| Class | Model ID | Provider |
|-------|----------|----------|
| `GPT4o` | `gpt-4o-2024-08-06` | OpenAI |
| `GPT4oMini` | `gpt-4o-mini-2024-07-18` | OpenAI |
| `Claude35Sonnet` | `claude-3-5-sonnet-20241022` | Anthropic |
| `Claude35Haiku` | `claude-3-5-haiku-20241022` | Anthropic |
| `MetaLlama` | `meta-llama/llama-3.3-70b-instruct` | OpenRouter |
| `Qwen` | `qwen/qwen-2.5-72b-instruct` | OpenRouter |
| `DeepSeek` | `deepseek/deepseek-chat` | OpenRouter |
| `MistralLarge` | `mistralai/mistral-large-2411` | OpenRouter |

**Usage:**
```python
from models.models import GPT4o, Claude35Sonnet

model = GPT4o(temperature=0.1)
response = model.invoke("What is 2+2?")
```

### Utility Functions (`utils/path.py`)

```python
def get_project_path() -> Path:
    """Returns absolute path to project root."""

def load_prompt(template_name: str) -> str:
    """Loads prompt from resource/prompts/{template_name}."""

def read_text_file(file_path: str) -> str:
    """Reads file content."""
```

### Config (`utils/config.py`)

```python
def get_config() -> Dict:
    """Loads and returns config from conf.d/config.yaml."""

def get_config_file_path() -> Path:
    """Returns path to config file."""
```

---

## 10. Supported Models & API Cost Estimation

### 10.1 Supported Models

TSG-Bench supports a wide range of models across multiple providers.

#### OpenAI Models

| Class | Model ID | Context Window |
|-------|----------|----------------|
| `GPT4o` | `gpt-4o-2024-08-06` | 128K |
| `GPT4oMini` | `gpt-4o-mini-2024-07-18` | 128K |
| `GPT41` | `gpt-4.1` | 1M |
| `GPT41Mini` | `gpt-4.1-mini` | 1M |
| `GPT41Nano` | `gpt-4.1-nano` | 1M |
| `GPT5` | `gpt-5` | 272K |
| `GPT5Mini` | `gpt-5-mini` | 272K |
| `GPT5Nano` | `gpt-5-nano` | 272K |
| `GPT5Pro` | `gpt-5-pro` | 272K |
| `O1` | `o1` | 128K |
| `O1Pro` | `o1-pro` | 128K |
| `O3` | `o3` | 200K |
| `O3Mini` | `o3-mini` | 200K |
| `O4Mini` | `o4-mini` | 200K |

#### Anthropic Models

| Class | Model ID | Context Window |
|-------|----------|----------------|
| `Claude35Sonnet` | `claude-3-5-sonnet-20241022` | 200K |
| `Claude35Haiku` | `claude-3-5-haiku-20241022` | 200K |

#### OpenRouter Models

| Class | Model ID |
|-------|----------|
| `MetaLlama` | `meta-llama/llama-3.3-70b-instruct` |
| `Qwen` | `qwen/qwen-2.5-72b-instruct` |
| `Qwen7B` | `qwen/qwen-2.5-7b-instruct` |
| `DeepSeek` | `deepseek/deepseek-chat` |
| `MistralMixtral` | `mistralai/mixtral-8x22b-instruct` |
| `MistralLarge` | `mistralai/mistral-large-2411` |
| `Mistral7B` | `mistralai/mistral-7b-instruct` |

### 10.2 API Pricing (Per Million Tokens)

*Prices as of November 2025. Verify at [OpenAI Pricing](https://openai.com/api/pricing/) for latest rates.*

#### OpenAI Models

| Model | Input | Output |
|-------|-------|--------|
| GPT-4o | $2.50 | $10.00 |
| GPT-4o-mini | $0.15 | $0.60 |
| GPT-4.1 | $2.00 | $8.00 |
| GPT-4.1-mini | $0.40 | $1.60 |
| GPT-4.1-nano | $0.10 | $0.40 |
| GPT-5 | $1.25 | $10.00 |
| GPT-5-mini | $0.25 | $2.00 |
| GPT-5-nano | $0.05 | $0.40 |
| GPT-5-pro | $15.00 | $120.00 |
| o1 | $15.00 | $60.00 |
| o1-pro | $150.00 | $600.00 |
| o3 | $10.00 | $40.00 |
| o3-mini | $1.10 | $4.40 |
| o4-mini | $1.10 | $4.40 |

#### Anthropic Models

| Model | Input | Output |
|-------|-------|--------|
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 3.5 Haiku | $0.80 | $4.00 |

### 10.3 SGQA Dataset Token Statistics

| Metric | Value |
|--------|-------|
| Total entries | 100 |
| Total QA pairs | 500 |
| Avg graphs per entry | 25.5 (range: 11-74) |
| Avg input tokens per request | ~2,683 |
| Avg output tokens per request | ~57 |
| **Total input tokens (full eval)** | ~268,366 |
| **Total output tokens (full eval)** | ~5,716 |

### 10.4 SGQA Cost Estimates Per Model

**Cost Formula:**
```
Cost = (input_tokens / 1M) × input_price + (output_tokens / 1M) × output_price
     = (268,366 / 1,000,000) × input_price + (5,716 / 1,000,000) × output_price
```

| Model | Estimated Cost | Category |
|-------|----------------|----------|
| GPT-5-nano | $0.02 | Budget |
| GPT-4.1-nano | $0.03 | Budget |
| GPT-4o-mini | $0.04 | Budget |
| GPT-5-mini | $0.08 | Budget |
| GPT-4.1-mini | $0.12 | Budget |
| Claude 3.5 Haiku | $0.24 | Budget |
| o3-mini | $0.32 | Budget |
| o4-mini | $0.32 | Budget |
| GPT-5 | $0.39 | Mid-tier |
| GPT-4.1 | $0.58 | Mid-tier |
| GPT-4o | $0.73 | Mid-tier |
| Claude 3.5 Sonnet | $0.89 | Mid-tier |
| o3 | $2.91 | Premium |
| o1 | $4.37 | Premium |
| GPT-5-pro | $4.71 | Premium |
| o1-pro | $43.68 | Premium |

### 10.5 Total Cost Summary

#### All OpenAI Models (14 models)

| Category | Cost Range |
|----------|------------|
| Budget (nano/mini) | $0.02 - $0.32 |
| Mid-tier (standard) | $0.39 - $2.91 |
| Premium (pro/o1) | $4.37 - $43.68 |
| **Total (all 14)** | **~$58.30** |

#### All Models (16 total)

| Provider | Models | Total Cost |
|----------|--------|------------|
| OpenAI | 14 | ~$58.30 |
| Anthropic | 2 | ~$1.13 |
| **Grand Total** | **16** | **~$59.43** |

### 10.6 Cost Optimization Tips

1. **Use Budget Models First**: GPT-5-nano ($0.02) and GPT-4.1-nano ($0.03) are cheapest for initial testing

2. **Prompt Caching**: Reduces costs by 50-90% depending on model
   - GPT-5 family: 90% discount on cached inputs
   - GPT-4.1 family: 75% discount
   - GPT-4o/o-series: 50% discount

3. **Batch API**: 50% discount for non-urgent workloads (24-hour processing)

4. **Avoid Reasoning Models for Simple Tasks**: o1-pro costs 100x more than GPT-5-nano

5. **Best Value Picks**:
   - Budget: GPT-4o-mini ($0.04) - excellent performance/cost ratio
   - Mid-tier: GPT-5 ($0.39) - good balance of capability and cost
   - Premium: o3 ($2.91) - reasoning capabilities at reasonable cost

### 10.7 Notes

- **Reasoning tokens**: o-series models may incur additional costs for internal reasoning (billed as output tokens but not visible in response)
- **Context limits**: Larger context windows may result in higher costs per request
- **Rate limits**: Different tiers have different rate limits; consider this for parallel evaluation

---

## 11. Running Evaluations

### SGQA Evaluation

```python
from models.models import GPT4o
from evaluation.understanding.sgqa import QAEvaluator, QADataLoader

# Initialize
model = GPT4o()
evaluator = QAEvaluator(model)
data_loader = QADataLoader()

# Run evaluation
results = evaluator.evaluate(data_loader)

print(f"Model: {results['model']}")
print(f"Accuracy: {results['accuracy']:.3f}")
print(f"Correct: {results['total_correct']}/{results['total_questions']}")
```

### SGDS Evaluation

```python
from models.models import Claude35Sonnet
from evaluation.understanding.sgds import SceneGraphEvaluator

model = Claude35Sonnet()
evaluator = SceneGraphEvaluator(model)

results = evaluator.evaluate_dataset("resource/dataset/understanding/sgds.jsonl")

print(f"Accuracy: {results['accuracy']:.3f}")
```

### SA-SGG Evaluation

```python
from models.models import GPT4o
from evaluation.generation.sasgg import GraphEvaluator

model = GPT4o()
evaluator = GraphEvaluator(model)

results = evaluator.evaluate("resource/dataset/generation/sa-sgg.jsonl")

print(f"Macro Precision: {results['macro_precision']:.3f}")
print(f"Macro Recall: {results['macro_recall']:.3f}")
print(f"Macro F1: {results['macro_f1']:.3f}")
```

### MA-SGG Evaluation

```python
from models.models import GPT4oMini
from evaluation.generation.masgg import GraphEvaluator

model = GPT4oMini(temperature=1.0)  # Higher temperature for diversity
evaluator = GraphEvaluator(model)

results = evaluator.evaluate("resource/dataset/generation/ma-sgg.jsonl")

print(f"Macro F1: {results['macro_f1']:.3f}")
```

### Adding a New Model

```python
# In models/models.py

class MyNewModel(LLM):
    def __init__(self, temperature=0.1):
        super().__init__()
        # Initialize your model client
        self.client = MyModelClient(api_key=self.config["my_provider"]["key"])
        self.temperature = temperature

    def invoke(self, message: str) -> str:
        response = self.client.generate(
            prompt=message,
            temperature=self.temperature
        )
        return response.text.strip()
```

### Command Line Evaluation

```bash
# Activate environment
conda activate tsg-bench

# Run specific evaluation script
python evaluation/understanding/sgqa.py
python evaluation/understanding/sgds.py
python evaluation/generation/sa-sgg.py
python evaluation/generation/ma-sgg.py
```

---

## Appendix: Quick Reference

### Dataset Paths

```
resource/dataset/generation/sa-sgg.jsonl    # 1,188 samples
resource/dataset/generation/ma-sgg.jsonl    # 853 samples
resource/dataset/understanding/sgqa.jsonl   # 100 samples (500 QA pairs)
resource/dataset/understanding/sgds.jsonl   # 250 samples
```

### Prompt Paths

```
resource/prompts/sa-sgg.txt
resource/prompts/ma-sgg.txt
resource/prompts/sgqa.txt
resource/prompts/sgds.txt
```

### Evaluation Scripts

```
evaluation/generation/sa-sgg.py
evaluation/generation/ma-sgg.py
evaluation/understanding/sgqa.py
evaluation/understanding/sgds.py
```

### Metrics Summary

| Task | Metric | Formula |
|------|--------|---------|
| SGQA | Accuracy | correct / total |
| SGDS | Accuracy | correct / total |
| SA-SGG | Macro P/R/F1 | mean(metric_i) |
| MA-SGG | Macro P/R/F1 | mean(metric_i) |

---

*Documentation generated for TSG-Bench repository.*
