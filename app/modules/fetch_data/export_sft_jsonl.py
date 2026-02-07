"""
Export locally downloaded SFT datasets (saved under app/data/sft) into jsonl files
compatible with the SFT TaskMixture (messages-format).

Usage:
    python -m app.modules.fetch_data.export_sft_jsonl \
        --data-dir app/data/sft \
        --out-dir app/data/sft/jsonl

Outputs (jsonl):
    smol_smoltalk_train.jsonl
    smol_smoltalk_test.jsonl
    mmlu_auxiliary_train_train.jsonl
    mmlu_all_test.jsonl
    gsm8k_main_train.jsonl
    gsm8k_main_test.jsonl
    (identity_conversations.jsonl copied if present)

Each line: {"messages": [...]} matching sft.py rendering.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

from datasets import load_from_disk

# ---- helpers to convert rows -> messages ----

LETTERS = ("A", "B", "C", "D")


def render_mc(question: str, choices: List[str]) -> str:
    prompt = f"Multiple Choice question: {question}\n"
    prompt += "".join([f"- {choice}={letter}\n" for letter, choice in zip(LETTERS, choices)])
    prompt += "\nRespond only with the letter of the correct answer."
    return prompt


def smol_row_to_conv(row: Dict[str, Any]) -> Dict[str, Any]:
    msgs = row["messages"]
    if msgs and msgs[0].get("role") == "system":
        msgs = msgs[1:]
    return {"messages": msgs}


def mmlu_row_to_conv(row: Dict[str, Any]) -> Dict[str, Any]:
    # Some saved splits wrap content under a 'train' key
    if "train" in row and isinstance(row["train"], dict):
        row = row["train"]
    question = row["question"]
    choices = row["choices"]
    ans = row["answer"]
    user_msg = render_mc(question, choices)
    assistant_msg = LETTERS[ans]
    return {"messages": [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]}


GSM_RE = re.compile(r"(<<[^>]+>>)")


def gsm8k_row_to_conv(row: Dict[str, Any]) -> Dict[str, Any]:
    question = row["question"]
    answer = row["answer"]
    parts = []
    for part in re.split(r"(<<[^>]+>>)", answer):
        if part.startswith("<<") and part.endswith(">>"):
            inner = part[2:-2]
            expr, result = inner.rsplit("=", 1) if "=" in inner else (inner, "")
            parts.append({"type": "python", "text": expr})
            if result:
                parts.append({"type": "python_output", "text": result})
        else:
            if part:
                parts.append({"type": "text", "text": part})
    return {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": parts},
    ]}


CONVERTERS = {
    "smol_smoltalk": smol_row_to_conv,
    "mmlu_auxiliary_train": mmlu_row_to_conv,
    "mmlu_all": mmlu_row_to_conv,
    "gsm8k_main": gsm8k_row_to_conv,
}


# ---- main ----


def export_split(name: str, split: str, data_dir: Path, out_dir: Path):
    ds_path = data_dir / name / split
    if not ds_path.exists():
        print(f"[skip] {ds_path} not found")
        return
    ds = load_from_disk(str(ds_path))
    convert = CONVERTERS[name]
    out_path = out_dir / f"{name}_{split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            conv = convert(row)
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {len(ds):,} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Export local SFT datasets to jsonl")
    parser.add_argument("--data-dir", type=str, default="app/data/sft", help="Directory containing downloaded datasets")
    parser.add_argument("--out-dir", type=str, default="app/data/sft/jsonl", help="Output directory for jsonl files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    combos = [
        ("smol_smoltalk", "train"),
        ("smol_smoltalk", "test"),
        ("mmlu_auxiliary_train", "train"),
        ("mmlu_all", "test"),
        ("gsm8k_main", "train"),
        ("gsm8k_main", "test"),
    ]

    for name, split in combos:
        export_split(name, split, data_dir, out_dir)

    # Copy identity if present
    identity = data_dir / "identity_conversations.jsonl"
    if identity.exists():
        dst = out_dir / "identity_conversations.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.resolve() != identity.resolve():
            dst.write_bytes(identity.read_bytes())
        print(f"[ok] copied identity -> {dst}")
    else:
        print("[info] identity_conversations.jsonl not found; skipped")


if __name__ == "__main__":
    main()
