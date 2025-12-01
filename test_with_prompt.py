"""
Show the complete prompt sent to the model
"""

import json
from langchain_core.prompts import PromptTemplate
from utils.path import load_prompt

def show_full_prompt():
    """Display the exact prompt sent to GPT-4o-mini"""

    print("=" * 80)
    print("COMPLETE PROMPT SENT TO MODEL")
    print("=" * 80)

    # Load first sample
    data_path = "resource/dataset/generation/sa-sgg.jsonl"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.loads(f.readline())

    # Load prompt template
    prompt_template_text = load_prompt("sa-sgg.txt")
    prompt_template = PromptTemplate(
        input_variables=[
            "context",
            "target_sentence",
            "available_nodes",
            "available_edges",
        ],
        template=prompt_template_text,
    )

    # Prepare data
    context = data["context"]
    target_sentence = data["target_sentence"]
    available_nodes = ", ".join(data["mandatory_space"]["object"])
    available_edges = ", ".join(data["mandatory_space"]["relationship"])
    verbs = data["mandatory_space"]["verb"]
    all_nodes = f"{available_nodes}, {', '.join(verbs)}"

    # Format the complete prompt
    full_prompt = prompt_template.format(
        context=context,
        target_sentence=target_sentence,
        available_nodes=all_nodes,
        available_edges=available_edges,
    )

    print(full_prompt)
    print("\n" + "=" * 80)
    print("END OF PROMPT")
    print("=" * 80)

if __name__ == "__main__":
    show_full_prompt()
