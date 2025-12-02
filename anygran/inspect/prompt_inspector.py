"""
Prompt Inspector for AnyGran SGQA
=================================
Generates and saves exact prompts for both baseline and hierarchical approaches
to help debug and understand model behavior.

Usage:
    python anygran/inspect/prompt_inspector.py --limit 10
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.prompts import PromptTemplate
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


def load_sgqa_data(limit: int = None) -> List[Dict]:
    """Load SGQA dataset and flatten QA pairs."""
    data_path = Path(__file__).parent.parent.parent / "resource" / "dataset" / "understanding" / "sgqa.jsonl"

    qa_data = []
    sample_count = 0

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                sample_count += 1

                for qa_pair in item["qa_pairs"]:
                    qa_data.append({
                        "data_id": item["data_id"],
                        "doc_index": item["doc_index"],
                        "text_part_index": item["text_part_index"],
                        "context_graphs": item["context_graphs"],
                        "question": qa_pair["Q"],
                        "answer": qa_pair["A"],
                    })

                if limit and sample_count >= limit:
                    break

    return qa_data


def format_sub_events(sub_events: List[Dict]) -> str:
    """Format sub-events list for the prompt."""
    if not sub_events:
        return "No sub-events available"

    lines = []
    for i, event in enumerate(sub_events):
        name = event.get("name", f"Phase {i+1}")
        desc = event.get("description", "")
        indices = event.get("action_indices", [])
        lines.append(f"{i+1}. {name}: {desc} (Actions: {indices})")

    return "\n".join(lines)


def generate_baseline_prompt(scene_graph: str, question: str) -> str:
    """Generate the baseline prompt (no hierarchy)."""
    template = """You are a highly advanced language model specialized in answering questions based on a given scene graph and question. Your task is to analyze the scene graph and provide the correct answer in a single word. Your output must strictly follow the format [answer], and nothing else should be printed. Ensure that your answer is concise, accurate, and matches the format exactly.

Scene Graph: {scene_graph}
Question: {question}
"""
    return template.format(scene_graph=scene_graph, question=question)


def generate_hierarchical_prompt(
    overall_goal: str,
    sub_events: List[Dict],
    scene_graph: str,
    question: str
) -> str:
    """Generate the hierarchical prompt (v0 - separated sections)."""
    prompt_template = load_prompt("hierarchical_sgqa.txt")
    sub_events_str = format_sub_events(sub_events)

    return prompt_template.format(
        overall_goal=overall_goal,
        sub_events=sub_events_str,
        scene_graph=scene_graph,
        question=question,
    )


def extract_answer(response: str) -> str:
    """Extract answer from [brackets]."""
    answer = re.findall(r"\[(.*?)\]", response)
    return answer[0] if answer else response.strip()


def save_inspection_file(
    output_dir: Path,
    filename: str,
    data_id: str,
    question: str,
    ground_truth: str,
    prompt: str,
    response: str,
    extracted_answer: str,
    is_correct: bool
):
    """Save a single inspection file."""
    content = f"""# Sample: {data_id}
# Question: {question}
# Ground Truth: {ground_truth}

---

## Full Prompt Sent to Model

```
{prompt}
```

---

## Model Response

```
{response}
```

---

## Extracted Answer
{extracted_answer}

## Result
- Correct: {"Yes ✓" if is_correct else "No ✗"}
- Ground Truth: {ground_truth}
- Prediction: {extracted_answer}
"""

    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Inspect SGQA prompts and responses")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of samples to process (default: 10)"
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path to cached event hierarchies JSON"
    )
    parser.add_argument(
        "--no-call",
        action="store_true",
        help="Don't call the model, just generate prompts"
    )

    args = parser.parse_args()

    # Load data
    print("Loading SGQA dataset...")
    data = load_sgqa_data(limit=args.limit)
    print(f"Loaded {len(data)} QA pairs from {args.limit} samples")

    # Load events cache
    cache_path = args.cache_path or str(Path(__file__).parent.parent / "cache" / "events_limitall.json")
    print(f"Loading events cache from: {cache_path}")
    events_cache = load_events_cache(cache_path)

    # Initialize model
    model = None
    if not args.no_call:
        print("Initializing GPT-5 model...")
        model = GPT5()

    # Create output directories
    base_output_dir = Path(__file__).parent

    # Group data by sample
    samples = {}
    for item in data:
        data_id = item["data_id"]
        if data_id not in samples:
            samples[data_id] = []
        samples[data_id].append(item)

    # Track results
    all_results = {
        "baseline": {"correct": 0, "total": 0, "results": []},
        "hierarchical": {"correct": 0, "total": 0, "results": []},
    }

    # Process each sample
    for sample_idx, (data_id, qa_pairs) in enumerate(samples.items()):
        sample_dir = base_output_dir / f"sample_{sample_idx+1:02d}_{data_id[:8]}"
        sample_dir.mkdir(exist_ok=True)

        print(f"\n[{sample_idx+1}/{len(samples)}] Processing sample {data_id[:8]}...")

        # Get event data for this sample
        event_data = events_cache.get(data_id, {})
        overall_goal = event_data.get("overall_goal", "Activity sequence")
        sub_events = event_data.get("sub_events", [])

        for q_idx, item in enumerate(qa_pairs):
            question = item["question"]
            ground_truth = item["answer"]
            scene_graph = str(item["context_graphs"])

            print(f"  Q{q_idx+1}: {question[:50]}...")

            # Generate baseline prompt
            baseline_prompt = generate_baseline_prompt(scene_graph, question)

            # Generate hierarchical prompt
            hier_prompt = generate_hierarchical_prompt(
                overall_goal, sub_events, scene_graph, question
            )

            if model and not args.no_call:
                # Call model for baseline
                baseline_response = model.invoke(baseline_prompt)
                baseline_answer = extract_answer(baseline_response)
                baseline_correct = baseline_answer.lower().strip() == ground_truth.lower().strip()

                # Call model for hierarchical
                hier_response = model.invoke(hier_prompt)
                hier_answer = extract_answer(hier_response)
                hier_correct = hier_answer.lower().strip() == ground_truth.lower().strip()

                # Track results
                all_results["baseline"]["total"] += 1
                all_results["hierarchical"]["total"] += 1
                if baseline_correct:
                    all_results["baseline"]["correct"] += 1
                if hier_correct:
                    all_results["hierarchical"]["correct"] += 1

                all_results["baseline"]["results"].append({
                    "data_id": data_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": baseline_answer,
                    "correct": baseline_correct,
                })
                all_results["hierarchical"]["results"].append({
                    "data_id": data_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": hier_answer,
                    "correct": hier_correct,
                })
            else:
                baseline_response = "[No model call - prompt only]"
                baseline_answer = "N/A"
                baseline_correct = False
                hier_response = "[No model call - prompt only]"
                hier_answer = "N/A"
                hier_correct = False

            # Save baseline inspection
            save_inspection_file(
                sample_dir,
                f"q{q_idx+1}_baseline.md",
                data_id,
                question,
                ground_truth,
                baseline_prompt,
                baseline_response,
                baseline_answer,
                baseline_correct
            )

            # Save hierarchical inspection
            save_inspection_file(
                sample_dir,
                f"q{q_idx+1}_hierarchical.md",
                data_id,
                question,
                ground_truth,
                hier_prompt,
                hier_response,
                hier_answer,
                hier_correct
            )

    # Generate summary
    if model and not args.no_call:
        generate_summary(base_output_dir, all_results, samples)

    print(f"\n{'='*60}")
    print("Inspection complete!")
    print(f"Output saved to: {base_output_dir}")
    if model and not args.no_call:
        baseline_em = (all_results["baseline"]["correct"] / all_results["baseline"]["total"]) * 100
        hier_em = (all_results["hierarchical"]["correct"] / all_results["hierarchical"]["total"]) * 100
        print(f"\nBaseline EM: {baseline_em:.1f}% ({all_results['baseline']['correct']}/{all_results['baseline']['total']})")
        print(f"Hierarchical EM: {hier_em:.1f}% ({all_results['hierarchical']['correct']}/{all_results['hierarchical']['total']})")


def generate_summary(output_dir: Path, results: Dict, samples: Dict):
    """Generate summary markdown file."""
    baseline_em = (results["baseline"]["correct"] / results["baseline"]["total"]) * 100
    hier_em = (results["hierarchical"]["correct"] / results["hierarchical"]["total"]) * 100

    # Find differences
    improvements = []
    regressions = []

    for i, (b_res, h_res) in enumerate(zip(results["baseline"]["results"], results["hierarchical"]["results"])):
        if not b_res["correct"] and h_res["correct"]:
            improvements.append({
                "question": b_res["question"],
                "ground_truth": b_res["ground_truth"],
                "baseline_pred": b_res["prediction"],
                "hier_pred": h_res["prediction"],
            })
        elif b_res["correct"] and not h_res["correct"]:
            regressions.append({
                "question": b_res["question"],
                "ground_truth": b_res["ground_truth"],
                "baseline_pred": b_res["prediction"],
                "hier_pred": h_res["prediction"],
            })

    content = f"""# SGQA Prompt Inspection Summary

Generated: {datetime.now().isoformat()}

## Overview

| Approach | Correct | Total | EM (%) |
|----------|---------|-------|--------|
| Baseline (GPT-5) | {results["baseline"]["correct"]} | {results["baseline"]["total"]} | {baseline_em:.1f}% |
| Hierarchical v0 (GPT-5) | {results["hierarchical"]["correct"]} | {results["hierarchical"]["total"]} | {hier_em:.1f}% |

## Improvements (Baseline ✗ → Hierarchical ✓)

"""

    if improvements:
        for imp in improvements:
            content += f"""### {imp["question"][:60]}...
- Ground Truth: **{imp["ground_truth"]}**
- Baseline Prediction: {imp["baseline_pred"]} ✗
- Hierarchical Prediction: {imp["hier_pred"]} ✓

"""
    else:
        content += "*No improvements*\n\n"

    content += """## Regressions (Baseline ✓ → Hierarchical ✗)

"""

    if regressions:
        for reg in regressions:
            content += f"""### {reg["question"][:60]}...
- Ground Truth: **{reg["ground_truth"]}**
- Baseline Prediction: {reg["baseline_pred"]} ✓
- Hierarchical Prediction: {reg["hier_pred"]} ✗

"""
    else:
        content += "*No regressions*\n\n"

    content += f"""## Samples Processed

Total samples: {len(samples)}
Total questions: {results["baseline"]["total"]}
"""

    summary_path = output_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
