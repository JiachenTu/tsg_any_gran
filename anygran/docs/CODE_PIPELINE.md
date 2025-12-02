# AnyGran Code Pipeline Explanation

This document explains where the key code logic is located in the AnyGran hierarchical SGQA pipeline.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ANYGRAN PIPELINE FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  run_hierarchical_sgqa.py (Main Entry Point)                                    │
│  ├── load_sgqa_data()           → Lines 70-96                                   │
│  │                                                                              │
│  ├── STEP 1: Event Generation   → Lines 226-235                                 │
│  │   └── EventGenerator (event_generator.py)                                    │
│  │                                                                              │
│  ├── STEP 2: Baseline Eval      → Lines 241-249                                 │
│  │   └── BaselineSGQAEvaluator (hierarchical_sgqa.py:192-287)                  │
│  │                                                                              │
│  └── STEP 3: Hierarchical Eval  → Lines 251-257                                 │
│      └── HierarchicalSGQAEvaluator (hierarchical_sgqa.py:49-189)               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## File 1: `run_hierarchical_sgqa.py` (Main Runner)

### Key Locations:

| Lines | Function | Description |
|-------|----------|-------------|
| 38-51 | `GPT5Mini` class | GPT-5-mini model wrapper |
| 54-67 | `GPT5` class | GPT-5 model wrapper |
| 70-96 | `load_sgqa_data()` | Loads SGQA dataset, flattens QA pairs |
| 161-274 | `main()` | Main orchestration function |
| 215-224 | Model selection | Chooses GPT5 or GPT5Mini based on `--model` flag |
| 226-235 | Event generation | Generates/loads hierarchical events |
| 241-249 | Baseline evaluation | Runs `BaselineSGQAEvaluator` |
| 251-257 | Hierarchical evaluation | Runs `HierarchicalSGQAEvaluator` |

### Data Loading (Lines 70-96):
```python
def load_sgqa_data(limit: int = None) -> List[Dict]:
    # Reads sgqa.jsonl
    # Flattens: each sample has multiple qa_pairs
    # Returns: list of {data_id, context_graphs, question, answer}
```

### Pipeline Orchestration (Lines 226-257):
```python
# STEP 1: Generate events (or load from cache)
generator = EventGenerator(model=event_model)
events_cache = generator.batch_generate(data, output_path=cache_path)

# STEP 2: Baseline evaluation (no hierarchy)
baseline_evaluator = BaselineSGQAEvaluator(model=model)
baseline_results = baseline_evaluator.evaluate(data)

# STEP 3: Hierarchical evaluation (with events_cache)
hier_evaluator = HierarchicalSGQAEvaluator(model=model, events_cache=events_cache)
hier_results = hier_evaluator.evaluate(data)
```

---

## File 2: `hierarchical_sgqa.py` (Evaluators)

### Class 1: `BaselineSGQAEvaluator` (Lines 192-287)

**Purpose**: Standard SGQA without hierarchical context

#### Prompt Injection (Lines 205-213):
```python
self.prompt_template = PromptTemplate(
    input_variables=["scene_graph", "question"],
    template="""You are a highly advanced language model...
Scene Graph: {scene_graph}
Question: {question}
""",
)
```
- **Prompt is HARDCODED** in the class (not loaded from file)
- Only 2 variables: `scene_graph`, `question`

#### Inference Logic (Lines 215-225):
```python
def invoke(self, scene_graph: str, question: str) -> str:
    # 1. Format prompt with scene_graph and question
    prompt = self.prompt_template.format(
        scene_graph=scene_graph,
        question=question,
    )
    # 2. Call LLM
    response = self.model.invoke(prompt)
    # 3. Extract answer from [brackets] using regex
    answer = re.findall(r"\[(.*?)\]", response)
    return answer[0] if answer else response.strip()
```

#### Evaluation Loop (Lines 245-287):
```python
def evaluate(self, data: List[Dict], max_workers: int = 5) -> Dict:
    # Uses ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(self.process_single_question, item): idx
            for idx, item in enumerate(data)
        }
    # Calculates: exact_match_percent = (correct / total) * 100
```

#### Exact Match Logic (Lines 234-235):
```python
is_correct = prediction.lower().strip() == data["answer"].lower().strip()
```
- Case-insensitive comparison
- Strips whitespace

---

### Class 2: `HierarchicalSGQAEvaluator` (Lines 49-189)

**Purpose**: SGQA with multi-granular hierarchical context

#### Prompt Loading (Lines 64-69):
```python
# Loads from external file: anygran/prompts/hierarchical_sgqa.txt
prompt = load_prompt("hierarchical_sgqa.txt")
self.prompt_template = PromptTemplate(
    input_variables=["overall_goal", "sub_events", "scene_graph", "question"],
    template=prompt,
)
```
- **Prompt loaded from file** (unlike baseline)
- 4 variables: `overall_goal`, `sub_events`, `scene_graph`, `question`

#### Prompt Injection (Lines 71-102):
```python
def invoke(self, data_id: str, scene_graph: str, question: str) -> str:
    # 1. Get event hierarchy from cache (generated by EventGenerator)
    event_data = self.events_cache.get(data_id, {})
    overall_goal = event_data.get("overall_goal", "Activity sequence")
    sub_events = event_data.get("sub_events", [])

    # 2. Format sub-events into readable string
    sub_events_str = self._format_sub_events(sub_events)

    # 3. Build full prompt with all 4 variables
    prompt = self.prompt_template.format(
        overall_goal=overall_goal,      # Level 3
        sub_events=sub_events_str,       # Level 2
        scene_graph=scene_graph,         # Level 1 & 0
        question=question,
    )

    # 4. Call LLM and extract answer
    response = self.model.invoke(prompt)
    answer = re.findall(r"\[(.*?)\]", response)
    return answer[0] if answer else response.strip()
```

#### Sub-Events Formatting (Lines 104-116):
```python
def _format_sub_events(self, sub_events: List[Dict]) -> str:
    # Converts: [{"name": "...", "description": "...", "action_indices": [...]}]
    # Into: "1. Phase name: description (Actions: [0, 1, 2])"
    lines = []
    for i, event in enumerate(sub_events):
        name = event.get("name", f"Phase {i+1}")
        desc = event.get("description", "")
        indices = event.get("action_indices", [])
        lines.append(f"{i+1}. {name}: {desc} (Actions: {indices})")
    return "\n".join(lines)
```

---

## File 3: `event_generator.py` (Hierarchy Generation)

### Class: `EventGenerator` (Lines 49-192)

**Purpose**: Generate hierarchical event summaries (Level 2 & 3) from raw scene graphs

#### Prompt Loading (Lines 62-67):
```python
# Loads from: anygran/prompts/event_generation.txt
prompt = load_prompt("event_generation.txt")
self.prompt_template = PromptTemplate(
    input_variables=["scene_graphs"],
    template=prompt,
)
```

#### Event Generation (Lines 69-97):
```python
def generate_event_hierarchy(self, context_graphs: List) -> Dict:
    # 1. Format scene graphs for readability
    formatted_graphs = self._format_graphs_for_prompt(context_graphs)

    # 2. Build prompt
    prompt = self.prompt_template.format(scene_graphs=formatted_graphs)

    # 3. Call LLM
    response = self.model.invoke(prompt)

    # 4. Parse JSON response
    event_data = self._parse_json_response(response)
    return event_data  # {"overall_goal": "...", "sub_events": [...]}
```

#### Graph Formatting (Lines 99-108):
```python
def _format_graphs_for_prompt(self, context_graphs: List) -> str:
    # Converts raw triplets into readable format:
    # "Action 0: pick-up
    #    ['pick-up', 'with', 'hand1']
    #    ['person', 'verb', 'pick-up']"
    lines = []
    for i, action_graph in enumerate(context_graphs):
        action_verb = self._extract_action_verb(action_graph)
        lines.append(f"Action {i}: {action_verb}")
        for triplet in action_graph:
            lines.append(f"  {triplet}")
    return "\n".join(lines)
```

---

## Prompt Files

### `prompts/hierarchical_sgqa.txt`:
```
You are a highly advanced language model specialized in answering questions
based on scene graphs with multi-level hierarchical context.

## Overall Goal (What is happening)
{overall_goal}                          ← Level 3

## Sub-Events (Phases of the activity)
{sub_events}                            ← Level 2

## Detailed Scene Graphs (Action-Level)
{scene_graph}                           ← Level 1 & 0

## Question
{question}

Your output must strictly follow the format [answer]...
```

### `prompts/event_generation.txt`:
```
Given a sequence of scene graphs, generate a hierarchical summary:
1. **Overall Goal**: A single sentence describing the main activity
2. **Sub-Events**: Group the actions into 2-5 logical phases

Scene Graphs: {scene_graphs}

Output format (JSON):
{
  "overall_goal": "...",
  "sub_events": [{"name": "...", "description": "...", "action_indices": [...]}]
}
```

---

## Key Code Logic Summary

| Component | File | Lines | Key Logic |
|-----------|------|-------|-----------|
| **Data Loading** | `run_hierarchical_sgqa.py` | 70-96 | Flatten QA pairs from JSONL |
| **Model Selection** | `run_hierarchical_sgqa.py` | 215-224 | GPT5 vs GPT5Mini based on `--model` |
| **Event Generation** | `event_generator.py` | 69-97 | LLM generates {goal, sub_events} |
| **Event Caching** | `event_generator.py` | 138-192 | Save/load to JSON file |
| **Baseline Prompt** | `hierarchical_sgqa.py` | 205-213 | Hardcoded, 2 variables |
| **Hierarchical Prompt** | `hierarchical_sgqa.py` | 64-69 | Loaded from file, 4 variables |
| **Answer Extraction** | Both evaluators | regex | `re.findall(r"\[(.*?)\]", response)` |
| **Exact Match** | Both evaluators | compare | `prediction.lower() == answer.lower()` |
| **Parallel Eval** | Both evaluators | ThreadPool | `concurrent.futures.ThreadPoolExecutor` |

---

## Execution Flow Diagram

```
                    run_hierarchical_sgqa.py
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   load_sgqa_data()   EventGenerator      Model Selection
        │                   │                   │
        │            ┌──────┴──────┐           │
        │            ▼             ▼           │
        │    event_generation.txt  LLM         │
        │            │             │           │
        │            └──────┬──────┘           │
        │                   ▼                  │
        │           events_cache.json          │
        │                   │                  │
        └───────────────────┼──────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    BaselineSGQAEvaluator          HierarchicalSGQAEvaluator
            │                               │
            ▼                               ▼
    Hardcoded prompt               hierarchical_sgqa.txt
    {scene_graph, question}        {overall_goal, sub_events,
            │                       scene_graph, question}
            ▼                               ▼
         LLM.invoke()                   LLM.invoke()
            │                               │
            ▼                               ▼
    re.findall(r"\[(.*?)\]")       re.findall(r"\[(.*?)\]")
            │                               │
            ▼                               ▼
    prediction.lower() == answer.lower() (Exact Match)
            │                               │
            └───────────────┬───────────────┘
                            ▼
                    print_comparison()
```
