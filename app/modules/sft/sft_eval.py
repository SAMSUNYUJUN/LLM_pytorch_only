"""
Simple SFT evaluation: compute validation bits-per-byte (bpb) on the jsonl val mix.
Usage:
    python -m app.modules.sft.sft_eval --ckpt-relpath base_checkpoints/d32/model_076800.pt
"""
import os
import json
import argparse
import random
import re
import torch
from contextlib import nullcontext

from app.modules.utils.utils import compute_init, compute_cleanup, autodetect_device_type, get_base_dir, print0, download_file_with_lock
from app.modules.utils.checkpoint_manager import build_model, find_last_step
from app.modules.tokenizer.tokenizer import get_token_bytes
from app.modules.model.loss_eval import evaluate_bpb


# Minimal task helpers (copied from sft.py to avoid importing its CLI parser)

class Task:
    def __len__(self):
        return self.num_examples()

    def num_examples(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.get_example(idx)

    def get_example(self, idx):
        raise NotImplementedError


class TaskMixture(Task):
    def __init__(self, tasks, seed=42):
        self.tasks = tasks
        self.index_map = []
        for ti, task in enumerate(tasks):
            for li in range(len(task)):
                self.index_map.append((ti, li))
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(len(self.index_map), generator=g).tolist()
        self.index_map = [self.index_map[i] for i in perm]

    def num_examples(self):
        return len(self.index_map)

    def get_example(self, idx):
        ti, li = self.index_map[idx]
        return self.tasks[ti][li]


class CustomJSON(Task):
    def __init__(self, filepath, stop=None):
        assert os.path.exists(filepath), f"JSONL not found: {filepath}"
        self.data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))
                if stop is not None and len(self.data) >= stop:
                    break

    def num_examples(self):
        return len(self.data)

    def get_example(self, idx):
        conv = self.data[idx]
        return {"messages": conv["messages"]} if "messages" in conv else conv


# Spelling tasks
LETTERS = "abcdefghijklmnopqrstuvwxyz"
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
TEST_RANDOM_SEED_OFFSET = 10_000_000
ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion: str):
    m = ANSWER_RE.search(completion)
    if m:
        return m.group(1).replace(",", "").strip()
    return None


class SpellingBee(Task):
    def __init__(self, size=1000, split="train"):
        assert split in ["train", "test"]
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, os.path.join("data", "sft", filename))
        with open(word_list_path, "r", encoding="utf-8") as f:
            self.words = [line.strip() for line in f]

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == "train" else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)
        word = rng.choice(self.words)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)
        count = word.count(letter)
        user_msg = f"How many {letter} are in the word {word}?"
        python_expr = f"'{word}'.count('{letter}')"
        assistant_parts = [
            {"type": "text", "text": f"Let's count in Python:"},
            {"type": "python", "text": python_expr},
            {"type": "python_output", "text": str(count)},
            {"type": "text", "text": f"\n#### {count}"},
        ]
        return {"messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_parts},
        ]}


class SimpleSpelling(Task):
    def __init__(self, size=1000, split="train"):
        assert split in ["train", "test"]
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, os.path.join("data", "sft", filename))
        with open(word_list_path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f]
        rng = random.Random(42)
        rng.shuffle(words)
        self.words = words

    def num_examples(self):
        return self.size

    def get_example(self, idx):
        seed = idx if self.split == "train" else TEST_RANDOM_SEED_OFFSET + idx
        rng = random.Random(seed)
        word = rng.choice(self.words)
        letters = ",".join(list(word))
        return {"messages": [
            {"role": "user", "content": f"Spell the word: {word}"},
            {"role": "assistant", "content": f"{word}:{letters}"},
        ]}


def get_sft_data_dir():
    return os.path.join(get_base_dir(), "data", "sft")


def maybe_path(path, base_dir):
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def build_val_dataset(jsonl_root, identity_path):
    def jp(name):
        return os.path.join(jsonl_root, name)
    val_tasks = [
        CustomJSON(jp("smol_smoltalk_test.jsonl")),
        CustomJSON(jp("mmlu_all_test.jsonl"), stop=5200),
        CustomJSON(jp("gsm8k_main_test.jsonl"), stop=420),
    ]
    if identity_path is not None and os.path.exists(identity_path):
        val_tasks.append(CustomJSON(identity_path))
    # include small spelling eval if desired (shortened)
    val_tasks.append(SimpleSpelling(size=1000, split="test"))
    val_tasks.append(SpellingBee(size=1000, split="test"))
    return TaskMixture(val_tasks, seed=7)


def batch_generator(val_dataset, tokenizer, device, max_seq_len, batch_size):
    bos = tokenizer.get_bos_token_id()
    idx = 0
    total = len(val_dataset)
    while True:
        batch_inputs = []
        batch_targets = []
        for _ in range(batch_size):
            conversation = val_dataset[idx % total]
            ids, mask = tokenizer.render_conversation(conversation, max_tokens=max_seq_len + 1)
            if len(ids) < 2:
                idx += 1
                continue
            ids = ids[: max_seq_len + 1]
            mask = mask[: max_seq_len + 1]
            # pad to max_seq_len+1
            pad_len = max_seq_len + 1 - len(ids)
            if pad_len > 0:
                ids = ids + [bos] * pad_len
                mask = mask + [0] * pad_len
            inputs = torch.tensor(ids[:-1], dtype=torch.int64)
            targets = torch.tensor(ids[1:], dtype=torch.int64)
            mask_t = torch.tensor(mask[1:], dtype=torch.int64)
            targets = torch.where(mask_t > 0, targets, torch.full_like(targets, -1))
            batch_inputs.append(inputs)
            batch_targets.append(targets)
            idx += 1
        inputs = torch.stack(batch_inputs).to(device=device)
        targets = torch.stack(batch_targets).to(device=device)
        yield inputs, targets


def main():
    parser = argparse.ArgumentParser(description="SFT eval: val bpb on jsonl val mix")
    parser.add_argument('--ckpt-relpath', type=str, required=True, help='Path under app/weights to checkpoint dir or model_XXXXXX.pt')
    parser.add_argument('--jsonl-dir', type=str, default=None, help='jsonl root (default: app/data/sft/jsonl)')
    parser.add_argument('--identity-jsonl', type=str, default=None, help='optional identity jsonl')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda','cpu','mps'], help='Device type (empty=autodetect)')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32','bfloat16'])
    parser.add_argument('--max-seq-len', type=int, default=2048)
    parser.add_argument('--device-batch-size', type=int, default=8)
    parser.add_argument('--eval-tokens', type=int, default=20*524288)
    args = parser.parse_args()

    base_dir = get_base_dir()
    jsonl_root = args.jsonl_dir or os.path.join(get_sft_data_dir(), "jsonl")
    jsonl_root = jsonl_root if os.path.isabs(jsonl_root) else os.path.join(base_dir, jsonl_root)

    identity_path = maybe_path(args.identity_jsonl, base_dir)
    if identity_path is None:
        default_identity = os.path.join(jsonl_root, "identity_conversations.jsonl")
        if os.path.exists(default_identity):
            identity_path = default_identity

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    weights_root = os.path.join(base_dir, "weights")
    full_path = os.path.join(weights_root, args.ckpt_relpath)
    if not os.path.exists(full_path):
        raise FileNotFoundError(full_path)
    if os.path.isfile(full_path):
        import re
        m = re.search(r"model_(\d+)\.pt", os.path.basename(full_path))
        if not m:
            raise ValueError(f"Cannot parse step from {full_path}")
        step = int(m.group(1))
        checkpoint_dir = os.path.dirname(full_path)
    else:
        checkpoint_dir = full_path
        step = find_last_step(checkpoint_dir)
    model, tokenizer, meta = build_model(checkpoint_dir, step, device, phase="eval", allow_missing_c_gate=True)
    model.eval()

    val_dataset = build_val_dataset(jsonl_root, identity_path)
    val_loader = batch_generator(val_dataset, tokenizer, device, args.max_seq_len, args.device_batch_size)

    eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * max(ddp_world_size,1)))
    token_bytes = get_token_bytes(device=device)
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
    print0(f"Validation bpb: {val_bpb:.4f}")

    compute_cleanup()

if __name__ == "__main__":
    main()
