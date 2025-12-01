# SGQA: Scene Graph Question Answering

## Overview

**SGQA (Scene Graph Question Answering)** is one of the four benchmark tasks in TSG-Bench. It evaluates an LLM's ability to understand spatio-temporal scene graphs and answer questions about the actions and relationships they represent.

## Task Description

Given a sequence of scene graphs representing a series of actions, the model must answer questions that require:
- **Temporal reasoning**: Understanding the order of events
- **Spatial reasoning**: Understanding object relationships and locations
- **Semantic understanding**: Interpreting action-object relationships

### Input
- A sequence of scene graphs (`context_graphs`)
- A natural language question

### Output
- A single-word or short-phrase answer

### Metric
- **Exact Match (EM)**: Case-insensitive string comparison between prediction and ground truth

---

## Dataset Format

**File**: `resource/dataset/understanding/sgqa.jsonl`

Each line is a JSON object with the following structure:

```json
{
  "data_id": "19cc4e42-39bb-41f9-b9de-9f2940eed6a2",
  "doc_index": 2,
  "text_part_index": 9,
  "context_graphs": [
    [["pick-up", "with", "hand1"], ["pick-up", "dobj", "mop-stick"], ["person", "verb", "pick-up"]],
    [["sweep", "dobj", "floor"], ["sweep", "with", "mop-stick"], ["person", "verb", "sweep"]],
    ...
  ],
  "qa_pairs": [
    {"Q": "What object was picked up before sweeping the floor?", "A": "mop-stick"},
    {"Q": "Which location did the person interact with after using the cloth?", "A": "wall"},
    ...
  ]
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `data_id` | string (UUID) | Unique identifier for this sample |
| `doc_index` | int | Document/scenario index |
| `text_part_index` | int | Index of the text part within the document |
| `context_graphs` | list[list[triplet]] | Sequence of scene graphs representing actions |
| `qa_pairs` | list[{Q, A}] | Question-Answer pairs about the scene graphs |

---

## Scene Graph Structure

Each action in `context_graphs` is represented as a list of triplets: `[node1, edge, node2]`

### Node Types
- **person**: The actor performing actions
- **action**: The verb/action being performed (e.g., pick-up, sweep, open)
- **object**: Items being manipulated (e.g., mop-stick, door, cloth)
- **hand**: Body part used (hand1, hand2)

### Edge Types (Relations)
- **verb**: Links person to action (`["person", "verb", "pick-up"]`)
- **dobj**: Direct object - links action to object (`["pick-up", "dobj", "mop-stick"]`)
- **with**: Instrument/tool used (`["pick-up", "with", "hand1"]`)
- **from/to/into/on**: Spatial relationships (`["put", "into", "box"]`)

### Example Scene Graph Sequence

```
Action 1: Person picks up mop-stick with hand1 and hand2
[
  ["pick-up", "with", "hand1"],
  ["pick-up", "with", "hand2"],
  ["mop-stick", "from", "floor"],
  ["person", "verb", "pick-up"],
  ["pick-up", "dobj", "mop-stick"]
]

Action 2: Person sweeps floor with mop-stick
[
  ["sweep", "with", "hand1"],
  ["sweep", "with", "hand2"],
  ["sweep", "with", "mop-stick"],
  ["sweep", "dobj", "floor"],
  ["sweep", "in", "car"],
  ["person", "verb", "sweep"]
]
```

---

## Question Types

SGQA questions test various reasoning capabilities:

### 1. Temporal Ordering
> "What object was picked up **before** sweeping the floor?"

Requires understanding the sequence of actions.

### 2. Causal Relationships
> "Which tool was picked up **after** the cooking vessel was placed on the heat source?"

Tests understanding of action dependencies.

### 3. Object Tracking
> "What object was handled **immediately after** opening the cabinet?"

Requires tracking objects across multiple actions.

### 4. Spatial Reasoning
> "Which item was placed **inside** the car after door interaction?"

Tests understanding of spatial relationships.

### 5. Action-Object Association
> "What was the **final object** picked from the wall?"

Requires identifying specific action-object pairs.

---

## Evaluation Code

### File: `run_sgqa_gpt.py`

The evaluation script follows this flow:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. LOAD DATASET                                             │
│    load_sgqa_data() reads sgqa.jsonl                        │
│    Flattens qa_pairs into individual samples                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. INITIALIZE MODEL                                         │
│    Creates LLM wrapper (GPT4o, GPT4oMini, etc.)            │
│    Loads API key from conf.d/config.yaml                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. CREATE EVALUATOR                                         │
│    SGQAEvaluator loads prompt from resource/prompts/sgqa.txt│
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. PARALLEL EVALUATION                                      │
│    For each QA pair:                                        │
│    a) Format prompt with scene_graph + question             │
│    b) Call model.invoke(prompt)                             │
│    c) Extract answer from [brackets] using regex            │
│    d) Compare with ground truth (case-insensitive)          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. CALCULATE METRICS                                        │
│    Exact Match % = (correct / total) × 100                 │
│    Save results to results/sgqa_{model}_{timestamp}.json   │
└─────────────────────────────────────────────────────────────┘
```

### Prompt Template

**File**: `resource/prompts/sgqa.txt`

```
You are a highly advanced language model specialized in answering questions
based on a given scene graph and question. Your task is to analyze the scene
graph and provide the correct answer in a single word. Your output must
strictly follow the format [answer], and nothing else should be printed.

Scene Graph: {scene_graph}
Question: {question}
```

### Key Code Components

```python
# Load and flatten dataset
def load_sgqa_data() -> List[Dict]:
    qa_data = []
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line)
            for qa_pair in item["qa_pairs"]:
                qa_data.append({
                    "data_id": item["data_id"],
                    "context_graphs": item["context_graphs"],
                    "question": qa_pair["Q"],
                    "answer": qa_pair["A"],
                })
    return qa_data

# Extract answer from model response
def invoke(self, scene_graph: str, question: str) -> str:
    prompt = self.prompt_template.format(
        scene_graph=scene_graph,
        question=question,
    )
    response = self.model.invoke(prompt)
    # Extract answer from [brackets]
    answer = re.findall(r"\[(.*?)\]", response)
    return answer[0] if answer else response

# Exact match comparison
is_correct = prediction.lower().strip() == data["answer"].lower().strip()
```

---

## Usage

### Command Line

```bash
# Evaluate with GPT-4o
python run_sgqa_gpt.py --model gpt4o

# Evaluate with GPT-4o-mini using 10 parallel workers
python run_sgqa_gpt.py --model gpt4o-mini --workers 10

# Evaluate all supported models
python run_sgqa_gpt.py --model all

# Run without saving results
python run_sgqa_gpt.py --model gpt4o --no-save
```

### Supported Models

| Flag | Model Class | API Provider |
|------|-------------|--------------|
| `gpt4o` | GPT4o | OpenAI |
| `gpt4o-mini` | GPT4oMini | OpenAI |
| `gpt41` | GPT41 | OpenAI |
| `gpt41-mini` | GPT41Mini | OpenAI |
| `gpt41-nano` | GPT41Nano | OpenAI |
| `o1` | O1 | OpenAI |
| `o1-pro` | O1Pro | OpenAI |
| `o3` | O3 | OpenAI |
| `o3-mini` | O3Mini | OpenAI |
| `o4-mini` | O4Mini | OpenAI |

---

## Output Format

Results are saved to `results/sgqa_{model}_{timestamp}.json`:

```json
{
  "task": "SGQA",
  "metric": "Exact Match (EM)",
  "timestamp": "2025-12-01T10:30:00",
  "model": "GPT4o",
  "total_questions": 500,
  "correct": 423,
  "exact_match_percent": 84.6,
  "results": [
    {
      "data_id": "19cc4e42-39bb-41f9-b9de-9f2940eed6a2",
      "question": "What object was picked up before sweeping the floor?",
      "ground_truth": "mop-stick",
      "prediction": "mop-stick",
      "exact_match": true
    },
    ...
  ]
}
```

---

## Dataset Statistics

- **Total samples**: ~500 unique scene graph sequences
- **QA pairs per sample**: ~5 questions each
- **Domains**: Cleaning, cooking, assembly, office work, etc.

---

## Related Files

| File | Description |
|------|-------------|
| `run_sgqa_gpt.py` | Main evaluation script for GPT models |
| `evaluation/understanding/sgqa.py` | Core SGQA evaluation module |
| `resource/dataset/understanding/sgqa.jsonl` | SGQA dataset |
| `resource/prompts/sgqa.txt` | Prompt template |
| `models/models.py` | LLM wrapper classes |
| `utils/path.py` | Path utilities and prompt loader |

---

## See Also

- [SGDS Task](./SGDS_TASK.md) - Scene Graph Description Selection
- [SA-SGG Task](./SA_SGG_TASK.md) - Single-Action Scene Graph Generation
- [MA-SGG Task](./MA_SGG_TASK.md) - Multi-Action Scene Graph Generation
