"""
Pre-download all SFT datasets to app/data/sft for offline training.

Usage:
    python -m app.modules.fetch_data.download_sft
    # or specify a target dir and identity file
    python -m app.modules.fetch_data.download_sft --target-dir app/data/sft --identity path/to/identity_conversations.jsonl
"""

import os
import shutil
import argparse

from datasets import load_dataset

from app.modules.utils.utils import get_base_dir, download_file_with_lock, print0

# reuse the same word list as spelling tasks
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"


def save_split(ds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.save_to_disk(path)
    print0(f"Saved split to {path}")


def download_sft(target_dir: str, identity_path: str | None):
    os.makedirs(target_dir, exist_ok=True)

    # SmolTalk
    smol_train = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    smol_test = load_dataset("HuggingFaceTB/smol-smoltalk", split="test")
    save_split(smol_train, os.path.join(target_dir, "smol_smoltalk", "train"))
    save_split(smol_test, os.path.join(target_dir, "smol_smoltalk", "test"))

    # MMLU: auxiliary_train/train + all/test
    mmlu_aux_train = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    mmlu_all_test = load_dataset("cais/mmlu", "all", split="test")
    save_split(mmlu_aux_train, os.path.join(target_dir, "mmlu_auxiliary_train", "train"))
    save_split(mmlu_all_test, os.path.join(target_dir, "mmlu_all", "test"))

    # GSM8K main train/test
    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    gsm_test = load_dataset("openai/gsm8k", "main", split="test")
    save_split(gsm_train, os.path.join(target_dir, "gsm8k_main", "train"))
    save_split(gsm_test, os.path.join(target_dir, "gsm8k_main", "test"))

    # Word list for spelling tasks
    filename = WORD_LIST_URL.split("/")[-1]
    download_file_with_lock(WORD_LIST_URL, os.path.join("data", "sft", filename))

    # Identity conversations (optional)
    if identity_path and os.path.exists(identity_path):
        dst = os.path.join(target_dir, "identity_conversations.jsonl")
        shutil.copy2(identity_path, dst)
        print0(f"Copied identity conversations to {dst}")
    else:
        print0("No identity_conversations.jsonl provided; skip.")

    print0("SFT dataset download complete.")


def main():
    parser = argparse.ArgumentParser(description="Download SFT datasets to local disk")
    parser.add_argument("--target-dir", type=str, default=None, help="Target directory (default: app/data/sft)")
    parser.add_argument("--identity", type=str, default=None, help="Optional identity_conversations.jsonl to copy")
    args = parser.parse_args()

    base_dir = get_base_dir()
    target_dir = args.target_dir or os.path.join(base_dir, "data", "sft")
    identity_path = args.identity
    if identity_path and not os.path.isabs(identity_path):
        identity_path = os.path.join(base_dir, identity_path)

    download_sft(target_dir, identity_path)


if __name__ == "__main__":
    main()
