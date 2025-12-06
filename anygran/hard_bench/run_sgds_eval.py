#!/usr/bin/env python3
"""Run SGDS Hard benchmark evaluation."""

import json
import re
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from utils.config import load_config, get_config_file_path


class GPT5:
    def __init__(self, temperature=0.1):
        config_path = get_config_file_path()
        config = load_config(config_path)
        self.openai = ChatOpenAI(
            api_key=config["openai"]["key"],
            model_name="gpt-5",
            temperature=temperature,
        )

    def invoke(self, message):
        return self.openai.invoke(message).content.strip()


class GPT5Mini:
    def __init__(self, temperature=0.1):
        config_path = get_config_file_path()
        config = load_config(config_path)
        self.openai = ChatOpenAI(
            api_key=config["openai"]["key"],
            model_name="gpt-5-mini",
            temperature=temperature,
        )

    def invoke(self, message):
        return self.openai.invoke(message).content.strip()


def main():
    # Load SGDS prompt
    prompt_path = Path(__file__).parent.parent.parent / "resource" / "prompts" / "sgds.txt"
    with open(prompt_path) as f:
        sgds_prompt = f.read()

    prompt_template = PromptTemplate(
        input_variables=["sentences", "triplet", "context"],
        template=sgds_prompt
    )

    # Load data
    cases = []
    with open(Path(__file__).parent / "sgds_hard.jsonl") as f:
        for line in f:
            cases.append(json.loads(line))

    print(f"SGDS Hard: {len(cases)} cases", flush=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_class, model_name in [(GPT5, "gpt5"), (GPT5Mini, "gpt5mini")]:
        print(f"\n=== Evaluating {model_name} ===", flush=True)
        model = model_class()
        results = []
        correct = 0

        for i, case in enumerate(cases):
            variations_str = "\n".join([
                f"{chr(65+j)}: {v}" for j, v in enumerate(case["variations"])
            ])

            prompt = prompt_template.format(
                sentences=variations_str,
                triplet=case["triplet"],
                context=case["context_graphs"]
            )

            response = model.invoke(prompt)
            match = re.search(r"\[([A-E])\]|\b([A-E])\b", response)
            pred = ord((match.group(1) or match.group(2))) - 65 if match else None
            is_correct = pred == case["position"] if pred is not None else False

            if is_correct:
                correct += 1

            results.append({
                "target_sentence": case["target_sentence"],
                "position": case["position"],
                "prediction": pred,
                "prediction_letter": chr(65 + pred) if pred is not None else None,
                "is_correct": is_correct,
                "hard_reason": case.get("hard_reason", ""),
                "context_length": case.get("context_length", 0),
            })

            print(f"[{model_name}] {i+1}/{len(cases)} | Acc: {correct/(i+1)*100:.1f}%", flush=True)

        output = {
            "task": "SGDS-Hard",
            "model": model_name,
            "timestamp": timestamp,
            "total_cases": len(cases),
            "correct": correct,
            "accuracy_percent": round(correct / len(cases) * 100, 2),
            "results": results,
        }

        output_path = Path(__file__).parent / "results" / f"sgds_hard_{model_name}_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved: {output_path}")

    print("\nSGDS evaluation complete!")


if __name__ == "__main__":
    main()
