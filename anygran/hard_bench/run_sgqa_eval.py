#!/usr/bin/env python3
"""Run SGQA Hard benchmark evaluation."""

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
    prompt_template = PromptTemplate(
        input_variables=["scene_graph", "question"],
        template="""You are a highly advanced language model specialized in answering questions based on a given scene graph and question. Your task is to analyze the scene graph and provide the correct answer in a single word. Your output must strictly follow the format [answer], and nothing else should be printed. Ensure that your answer is concise, accurate, and matches the format exactly.

Scene Graph: {scene_graph}
Question: {question}
""",
    )

    # Load data
    with open(Path(__file__).parent / "sgqa_hard.json") as f:
        cases = json.load(f)["cases"]

    print(f"SGQA Hard: {len(cases)} cases", flush=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_class, model_name in [(GPT5, "gpt5"), (GPT5Mini, "gpt5mini")]:
        print(f"\n=== Evaluating {model_name} ===", flush=True)
        model = model_class()
        results = []
        correct = 0

        for i, case in enumerate(cases):
            prompt = prompt_template.format(
                scene_graph=str(case["context_graphs"]),
                question=case["question"]
            )
            response = model.invoke(prompt)
            answer = re.findall(r"\[(.*?)\]", response)
            pred = answer[0] if answer else response.strip()
            is_correct = pred.lower().strip() == case["ground_truth"].lower().strip()

            if is_correct:
                correct += 1

            results.append({
                "data_id": case["data_id"],
                "question": case["question"],
                "ground_truth": case["ground_truth"],
                "prediction": pred,
                "exact_match": is_correct,
                "difficulty": case.get("difficulty", ""),
                "error_category": case.get("error_category", ""),
            })

            print(f"[{model_name}] {i+1}/{len(cases)} | EM: {correct/(i+1)*100:.1f}%", flush=True)

        output = {
            "task": "SGQA-Hard",
            "model": model_name,
            "timestamp": timestamp,
            "total_cases": len(cases),
            "correct": correct,
            "accuracy_percent": round(correct / len(cases) * 100, 2),
            "results": results,
        }

        output_path = Path(__file__).parent / "results" / f"sgqa_hard_{model_name}_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved: {output_path}")

    print("\nSGQA evaluation complete!")


if __name__ == "__main__":
    main()
