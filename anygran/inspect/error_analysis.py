"""
Error Analysis for SGQA
=======================
Generates detailed analysis for all incorrect cases across Baseline, Hier v0, and Unified v1.

Usage:
    python anygran/inspect/error_analysis.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from utils.config import load_config, get_config_file_path
from event_generator import load_events_cache


class GPT5:
    """GPT-5 model wrapper."""
    def __init__(self, temperature=0.1):
        config_path = get_config_file_path()
        config = load_config(config_path)
        self.openai = ChatOpenAI(
            api_key=config["openai"]["key"],
            model_name="gpt-5",
            temperature=temperature,
        )

    def invoke(self, message):
        response = self.openai.invoke(message)
        return response.content.strip()


def load_prompt(prompt_name: str) -> str:
    """Load prompt template from anygran/prompts directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / prompt_name
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_sgqa_data(limit: int = None) -> Dict[str, Dict]:
    """Load SGQA dataset and return dict keyed by data_id."""
    data_path = Path(__file__).parent.parent.parent / "resource" / "dataset" / "understanding" / "sgqa.jsonl"

    samples = {}
    sample_count = 0

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                samples[item["data_id"]] = item
                sample_count += 1
                if limit and sample_count >= limit:
                    break

    return samples


def extract_action_verb(action_graph: List) -> str:
    """Extract the main action verb from an action graph."""
    for triplet in action_graph:
        if len(triplet) >= 3 and triplet[1] in ["verb", "verbs"]:
            return triplet[2]
    return "unknown"


def format_scene_graphs_readable(context_graphs: List) -> str:
    """Format scene graphs in a human-readable way."""
    lines = []
    for i, action_graph in enumerate(context_graphs):
        verb = extract_action_verb(action_graph)
        lines.append(f"Action {i}: {verb}")
        for triplet in action_graph:
            lines.append(f"  {triplet}")
        lines.append("")
    return "\n".join(lines)


def format_sub_events(sub_events: List[Dict]) -> str:
    """Format sub-events for hierarchical prompt."""
    if not sub_events:
        return "No sub-events available"

    lines = []
    for i, event in enumerate(sub_events):
        name = event.get("name", f"Phase {i+1}")
        desc = event.get("description", "")
        indices = event.get("action_indices", [])
        lines.append(f"{i+1}. {name}: {desc} (Actions: {indices})")

    return "\n".join(lines)


def format_unified_timeline(sub_events: List[Dict], context_graphs: List) -> str:
    """Format unified/interleaved timeline for v1 prompt."""
    if not sub_events:
        lines = ["### All Actions"]
        for i, action_graph in enumerate(context_graphs):
            verb = extract_action_verb(action_graph)
            triplets_str = " ".join(str(t) for t in action_graph)
            lines.append(f"- Action {i} ({verb}): {triplets_str}")
        return "\n".join(lines)

    lines = []
    for phase_idx, event in enumerate(sub_events):
        name = event.get("name", f"Phase {phase_idx+1}")
        desc = event.get("description", "")
        action_indices = event.get("action_indices", [])

        lines.append(f"### Phase {phase_idx+1}: {name}")
        lines.append(f"{desc}")
        lines.append("")
        lines.append("Actions in this phase:")

        for action_idx in action_indices:
            if action_idx < len(context_graphs):
                action_graph = context_graphs[action_idx]
                verb = extract_action_verb(action_graph)
                triplets_str = " ".join(str(t) for t in action_graph)
                lines.append(f"- Action {action_idx} ({verb}): {triplets_str}")

        lines.append("")

    return "\n".join(lines)


def generate_baseline_prompt(scene_graph: str, question: str) -> str:
    """Generate the baseline prompt."""
    return f"""You are a highly advanced language model specialized in answering questions based on a given scene graph and question. Your task is to analyze the scene graph and provide the correct answer in a single word. Your output must strictly follow the format [answer], and nothing else should be printed. Ensure that your answer is concise, accurate, and matches the format exactly.

Scene Graph: {scene_graph}
Question: {question}
"""


def generate_hierarchical_prompt(overall_goal: str, sub_events: List[Dict], scene_graph: str, question: str) -> str:
    """Generate the hierarchical v0 prompt."""
    prompt_template = load_prompt("hierarchical_sgqa.txt")
    sub_events_str = format_sub_events(sub_events)

    return prompt_template.format(
        overall_goal=overall_goal,
        sub_events=sub_events_str,
        scene_graph=scene_graph,
        question=question,
    )


def generate_unified_prompt(overall_goal: str, sub_events: List[Dict], context_graphs: List, question: str) -> str:
    """Generate the unified v1 prompt."""
    prompt_template = load_prompt("unified_hierarchical.txt")
    unified_timeline = format_unified_timeline(sub_events, context_graphs)

    return prompt_template.format(
        overall_goal=overall_goal,
        unified_timeline=unified_timeline,
        question=question,
    )


def extract_answer(response: str) -> str:
    """Extract answer from [brackets]."""
    answer = re.findall(r"\[(.*?)\]", response)
    return answer[0] if answer else response.strip()


# Define the 9 error cases
ERROR_CASES = [
    {
        "id": 1,
        "data_id": "19cc4e42-39bb-41f9-b9de-9f2940eed6a2",
        "question": "Which location did the person interact with after using the cloth?",
        "ground_truth": "wall",
        "short_name": "cloth_wall",
        "category": "Temporal Ordering + Object/Location Confusion",
    },
    {
        "id": 2,
        "data_id": "fbf4150a-27d2-48a4-956f-b4f85ecde465",
        "question": "What was the first object the person interacted with after placing the laptop?",
        "ground_truth": "charger",
        "short_name": "laptop_charger",
        "category": "Temporal Ordering",
    },
    {
        "id": 3,
        "data_id": "954c2f61-64ad-4c89-a26f-ec4547a65fab",
        "question": "What object was handled immediately before being transferred to the cooking vessel?",
        "ground_truth": "onion",
        "short_name": "onion_pot",
        "category": "Temporal Ordering",
    },
    {
        "id": 4,
        "data_id": "954c2f61-64ad-4c89-a26f-ec4547a65fab",
        "question": "Which object was interacted with between two water-related actions?",
        "ground_truth": "pot",
        "short_name": "water_pot",
        "category": "Temporal Ordering + Object Confusion",
    },
    {
        "id": 5,
        "data_id": "277b18de-4ad9-4c09-970b-91fcea05097d",
        "question": "Which object required both hands to manipulate after all tools were stored?",
        "ground_truth": "wheel",
        "short_name": "wheel",
        "category": "Complex Multi-Step Reasoning",
    },
    {
        "id": 6,
        "data_id": "e9be1118-a5cf-4431-b2e8-e3edcfa9f949",
        "question": "What was the last tool picked up before the final positioning action?",
        "ground_truth": "screw",
        "short_name": "screw",
        "category": "Similar Object Disambiguation",
    },
    {
        "id": 7,
        "data_id": "860980fb-f992-4bb1-8a46-b644f58090e2",
        "question": "Which object required both hands for its first manipulation after being picked up?",
        "ground_truth": "metal-board",
        "short_name": "metal_board",
        "category": "Similar Object Disambiguation + Temporal",
    },
    {
        "id": 8,
        "data_id": "e9be1118-a5cf-4431-b2e8-e3edcfa9f949",
        "question": "What tool was picked up immediately after the first clamp was removed?",
        "ground_truth": "wood",
        "short_name": "clamp_wood",
        "category": "Temporal Ordering",
    },
    {
        "id": 9,
        "data_id": "e9be1118-a5cf-4431-b2e8-e3edcfa9f949",
        "question": "Which object was held in hand2 while the drill was being operated?",
        "ground_truth": "wood",
        "short_name": "drill_wood",
        "category": "Concurrent State Tracking",
    },
]


def generate_error_analysis(error: Dict, sample: Dict, events_cache: Dict, model: GPT5) -> str:
    """Generate detailed analysis for one error case."""
    data_id = error["data_id"]
    question = error["question"]
    ground_truth = error["ground_truth"]
    context_graphs = sample["context_graphs"]

    # Get event data
    event_data = events_cache.get(data_id, {})
    overall_goal = event_data.get("overall_goal", "Activity sequence")
    sub_events = event_data.get("sub_events", [])

    # Format scene graphs for prompts
    scene_graph_str = str(context_graphs)
    scene_graph_readable = format_scene_graphs_readable(context_graphs)

    # Generate prompts
    baseline_prompt = generate_baseline_prompt(scene_graph_str, question)
    hier_prompt = generate_hierarchical_prompt(overall_goal, sub_events, scene_graph_str, question)
    unified_prompt = generate_unified_prompt(overall_goal, sub_events, context_graphs, question)

    # Call model
    print(f"  Calling GPT-5 for baseline...")
    baseline_response = model.invoke(baseline_prompt)
    baseline_answer = extract_answer(baseline_response)

    print(f"  Calling GPT-5 for hierarchical v0...")
    hier_response = model.invoke(hier_prompt)
    hier_answer = extract_answer(hier_response)

    print(f"  Calling GPT-5 for unified v1...")
    unified_response = model.invoke(unified_prompt)
    unified_answer = extract_answer(unified_response)

    # Determine correctness
    baseline_correct = baseline_answer.lower().strip() == ground_truth.lower().strip()
    hier_correct = hier_answer.lower().strip() == ground_truth.lower().strip()
    unified_correct = unified_answer.lower().strip() == ground_truth.lower().strip()

    # Format events
    events_str = f"**Overall Goal**: {overall_goal}\n\n**Sub-Events**:\n"
    for i, se in enumerate(sub_events):
        events_str += f"{i+1}. {se.get('name', 'Phase')}: {se.get('description', '')} (Actions: {se.get('action_indices', [])})\n"

    # Generate markdown
    md = f"""# Error {error['id']}: {error['short_name']}

## Question
> {question}

## Summary
| Approach | Prediction | Correct? |
|----------|------------|----------|
| Ground Truth | **{ground_truth}** | - |
| Baseline | {baseline_answer} | {"✓" if baseline_correct else "✗"} |
| Hierarchical v0 | {hier_answer} | {"✓" if hier_correct else "✗"} |
| Unified v1 | {unified_answer} | {"✓" if unified_correct else "✗"} |

**Category**: {error['category']}
**Sample ID**: `{data_id}`

---

## Raw Input Data

### Scene Graphs (context_graphs)
```
{scene_graph_readable}
```

### Events Cache
{events_str}

---

## Prompts and Responses

### Baseline Prompt
```
{baseline_prompt}
```

### Baseline Response
```
{baseline_response}
```
**Extracted Answer**: {baseline_answer}

---

### Hierarchical v0 Prompt
```
{hier_prompt}
```

### Hierarchical v0 Response
```
{hier_response}
```
**Extracted Answer**: {hier_answer}

---

### Unified v1 Prompt
```
{unified_prompt}
```

### Unified v1 Response
```
{unified_response}
```
**Extracted Answer**: {unified_answer}

---

## Error Analysis

### Why is this question hard?
- **Question Type**: {error['category']}
- **Key Challenge**: Requires understanding {"temporal sequence" if "Temporal" in error['category'] else "object relationships"}

### What would the model need to do correctly?
1. Parse the scene graphs to identify relevant actions
2. {"Track temporal ordering" if "Temporal" in error['category'] else "Distinguish between similar objects"}
3. Map the question keywords to specific actions/objects

### Does the scene graph support the ground truth?
*[Requires manual verification by examining the action sequence above]*

"""
    return md


def main():
    print("Loading data...")
    samples = load_sgqa_data(limit=10)

    cache_path = Path(__file__).parent.parent / "cache" / "events_limitall.json"
    events_cache = load_events_cache(str(cache_path))

    print("Initializing GPT-5...")
    model = GPT5()

    output_dir = Path(__file__).parent / "errors"
    output_dir.mkdir(exist_ok=True)

    summary_lines = [
        "# Error Analysis Summary",
        "",
        "## Overview",
        "",
        "| # | Question | GT | Baseline | Hier v0 | Unified v1 | Category |",
        "|---|----------|----|---------:|--------:|-----------:|----------|",
    ]

    for error in ERROR_CASES:
        print(f"\nProcessing Error {error['id']}: {error['short_name']}...")

        sample = samples.get(error["data_id"])
        if not sample:
            print(f"  WARNING: Sample {error['data_id']} not found!")
            continue

        md_content = generate_error_analysis(error, sample, events_cache, model)

        output_file = output_dir / f"error_{error['id']:02d}_{error['short_name']}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"  Saved to: {output_file}")

    # Generate summary
    summary_file = output_dir / "SUMMARY.md"

    summary_content = """# Error Analysis Summary

## Overview

All 9 unique error cases across Baseline, Hierarchical v0, and Unified v1 approaches.

| # | Question | GT | Baseline | Hier v0 | Unified v1 | Category |
|---|----------|----|---------:|--------:|-----------:|----------|
| 1 | Which location after using cloth? | wall | cabinet ✗ | cabinet ✗ | cabinet ✗ | Temporal + Location |
| 2 | First object after placing laptop? | charger | laptop ✗ | laptop ✗ | laptop ✗ | Temporal |
| 3 | Handled before cooking vessel? | onion | pot ✗ | pot ✗ | pot ✗ | Temporal |
| 4 | Between two water actions? | pot | onion ✗ | onion ✗ | onion ✗ | Temporal + Object |
| 5 | Both hands after tools stored? | wheel | lid ✗ | wheel ✓ | wheel ✓ | Multi-Step |
| 6 | Last tool before final positioning? | screw | screwdriver ✗ | screwdriver ✗ | screwdriver ✗ | Disambiguation |
| 7 | Both hands first manipulation? | metal-board | paper ✗ | paper+cardboard ✗ | paper ✗ | Disambiguation |
| 8 | After first clamp removed? | wood | wood ✓ | clamp ✗ | wood ✓ | Temporal |
| 9 | Held in hand2 during drill? | wood | wood ✓ | wood ✓ | wood+piece ✗ | Concurrent State |

## Error Categories

### Category A: Temporal Ordering (Errors 1, 2, 3, 6, 8)
Questions requiring understanding of action sequence (before/after relationships).
- Common failure: Incorrect identification of temporal boundaries
- All approaches struggle with these

### Category B: Object/Location Confusion (Errors 1, 4)
Questions where model confuses similar concepts or object types.
- Error 1: Predicts "cabinet" (object) instead of "wall" (location)
- Error 4: Swaps "pot" and "onion"

### Category C: Similar Object Disambiguation (Errors 6, 7)
Questions requiring distinction between similar objects.
- Error 6: "screw" vs "screwdriver" - very similar names
- Error 7: "metal-board" vs "paper" - both flat objects

### Category D: Complex Reasoning (Error 5)
Requires multi-step inference.
- "after all tools were stored" requires tracking tool storage completion
- Hierarchical approaches help here (both v0 and v1 correct)

### Category E: Concurrent State Tracking (Error 9)
Requires tracking simultaneous states.
- What is held in hand2 WHILE drill operates
- Unified v1 over-specifies ("wood and wood-piece")

## Key Insights

1. **Temporal reasoning is the biggest challenge**: 5 of 9 errors involve temporal ordering
2. **Hierarchical context helps some cases**: Error 5 solved by v0 and v1 (not baseline)
3. **v1 regression on Error 9**: Over-specification issue with unified format
4. **v0 regression on Error 8**: Separated format caused confusion

## Recommendations

1. **Add explicit temporal markers** in prompts (e.g., "Action 0 happens BEFORE Action 1")
2. **Clarify object vs location** in question understanding
3. **Avoid over-specification** in unified format (simpler output format)
4. **Consider chain-of-thought** for multi-step reasoning
"""

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_content)

    print(f"\nSummary saved to: {summary_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
