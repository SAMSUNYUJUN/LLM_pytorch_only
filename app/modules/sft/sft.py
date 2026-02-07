"""
Supervised fine-tuning (SFT) for the LLM_pytorch_only project.

Usage (multi-GPU):
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 app/modules/sft/sft.py --device-type cuda  --identity-jsonl app/data/sft/jsonl/identity_conversations.jsonl --total-batch-size 524288 --device-batch-size 16

Usage (single GPU for smoke tests):
    python app/modules/sft/sft.py --device-type cuda --device-batch-size 2 --total-batch-size 8192 --num-iterations 20
"""

import os
import sys
import json
import time
import argparse
import random
import re
from typing import List, Tuple, Dict, Any

# Reduce CUDA OOM fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Imports with fallback for direct invocation
# -----------------------------------------------------------------------------
try:
    from ..utils.utils import (
        compute_init,
        compute_cleanup,
        autodetect_device_type,
        get_base_dir,
        print0,
        download_file_with_lock,
    )
    from ..utils.checkpoint_manager import (
        build_model,
        find_last_step,
        find_largest_model,
        save_checkpoint,
    )
    from ..tokenizer.tokenizer import get_tokenizer, get_token_bytes
    from ..model.loss_eval import evaluate_bpb
except ImportError:
    # Allow running the script directly (without -m) from repo root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from app.modules.utils.utils import (
        compute_init,
        compute_cleanup,
        autodetect_device_type,
        get_base_dir,
        print0,
        download_file_with_lock,
    )
    from app.modules.utils.checkpoint_manager import (
        build_model,
        find_last_step,
        find_largest_model,
        save_checkpoint,
    )
    from app.modules.tokenizer.tokenizer import get_tokenizer, get_token_bytes
    from app.modules.model.loss_eval import evaluate_bpb

# -----------------------------------------------------------------------------
# Small task abstractions (lightweight clone of nanochat/tasks)
# -----------------------------------------------------------------------------


class Task:
    """Minimal task wrapper that exposes __len__ and __getitem__."""

    def __len__(self):
        return self.num_examples()

    def num_examples(self):
        raise NotImplementedError

    def __getitem__(self, idx: int):
        return self.get_example(idx)

    def get_example(self, idx: int):
        raise NotImplementedError


class TaskMixture(Task):
    """Deterministically shuffles and mixes multiple tasks."""

    def __init__(self, tasks: List[Task], seed: int = 42):
        self.tasks = tasks
        self.index_map: List[Tuple[int, int]] = []
        for task_idx, task in enumerate(tasks):
            for local_idx in range(len(task)):
                self.index_map.append((task_idx, local_idx))
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(len(self.index_map), generator=g).tolist()
        self.index_map = [self.index_map[i] for i in perm]

    def num_examples(self):
        return len(self.index_map)

    def get_example(self, idx: int):
        task_idx, local_idx = self.index_map[idx]
        return self.tasks[task_idx][local_idx]


def render_mc(question: str, letters: Tuple[str, ...], choices: List[str]) -> str:
    """Render a multiple-choice question (matches nanochat formatting)."""
    prompt = f"Multiple Choice question: {question}\n"
    prompt += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    prompt += "\nRespond only with the letter of the correct answer."
    return prompt


def get_sft_data_dir():
    return os.path.join(get_base_dir(), "data", "sft")



class CustomJSON(Task):
    """Local jsonl conversations with a 'messages' field."""

    def __init__(self, filepath: str, stop: int | None = None):
        assert os.path.exists(filepath), f"JSONL file not found: {filepath}"
        self.filepath = filepath
        self.data: List[Dict[str, Any]] = []
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

    def get_example(self, idx: int):
        conv = self.data[idx]
        assert "messages" in conv, f"Missing 'messages' in line {idx} of {self.filepath}"
        return {"messages": conv["messages"]}


# -----------------------------------------------------------------------------
# Spelling tasks (full version from nanochat/tasks/spellingbee.py)
# -----------------------------------------------------------------------------

LETTERS = "abcdefghijklmnopqrstuvwxyz"
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
TEST_RANDOM_SEED_OFFSET = 10_000_000

ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion: str) -> str | None:
    match = ANSWER_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}",
    # Spanish
    "¿Cuántas {letter} hay en {word}?",
    "¿Cuántas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "¿Cuántas letras {letter} tiene {word}?",
    # Chinese (Simplified)
    "{word}中有多少个{letter}",
    "{word}里有几个{letter}",
    "数一下{word}中的{letter}",
    "{word}这个词里有多少{letter}",
    # Korean
    "{word}에 {letter}가 몇 개 있나요",
    "{word}에서 {letter}의 개수는",
    "{word}에 {letter}가 몇 번 나오나요",
    "{word}라는 단어에 {letter}가 몇 개",
    # French
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} apparaît dans {word}",
    "Compte les {letter} dans {word}",
    # German
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
    # Japanese
    "{word}に{letter}は何個ありますか",
    "{word}の中に{letter}がいくつ",
    "{word}に{letter}が何回出てくる",
]


class SpellingBee(Task):
    """Count occurrences of a letter in a word, with reasoning + python tool use."""

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f]
        self.words = words

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == "train" else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)

        word = rng.choice(self.words)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)
        count = word.count(letter)

        template = rng.choice(USER_MSG_TEMPLATES)
        if rng.random() < 0.3:
            template = template.lower()
        quote_options = ["", "'", '"']
        letter_quote = rng.choice(quote_options)
        word_quote = rng.choice(quote_options)
        letter_wrapped = f"{letter_quote}{letter}{letter_quote}"
        word_wrapped = f"{word_quote}{word}{word_quote}"
        user_msg = template.format(letter=letter_wrapped, word=word_wrapped)
        if rng.random() < 0.5:
            user_msg += "?"

        assistant_parts = []
        word_letters = ",".join(list(word))
        manual_text = f"""We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.

First spell the word out:
{word}:{word_letters}

Then count the occurrences of '{letter}':
"""
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                manual_text += f"{i}:{char} hit! count={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"
        manual_text += f"\nThis gives us {running_count}."
        assistant_parts.append({"type": "text", "text": manual_text})
        assistant_parts.append({"type": "text", "text": "\n\nLet me double check this using Python:\n\n"})
        python_expr = f"'{word}'.count('{letter}')"
        assistant_parts.append({"type": "python", "text": python_expr})
        assistant_parts.append({"type": "python_output", "text": str(count)})
        assistant_parts.append({"type": "text", "text": f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}"})

        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_parts},
        ]
        return {"messages": messages}

    def evaluate(self, conversation, assistant_response):
        assert isinstance(assistant_response, str)
        assistant_message = conversation["messages"][-1]
        assert assistant_message["role"] == "assistant"
        assert isinstance(assistant_message["content"], list)
        last_text_part = assistant_message["content"][-1]["text"]
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        return int(pred_num == ref_num)

    def reward(self, conversation, assistant_response):
        return float(self.evaluate(conversation, assistant_response))


class SimpleSpelling(Task):
    """Simpler task: just spell the word."""

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SimpleSpelling split must be train|test"
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f]
        rng = random.Random(42)
        rng.shuffle(words)
        self.words = words

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == "train" else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)
        word = rng.choice(self.words)
        word_letters = ",".join(list(word))
        messages = [
            {"role": "user", "content": f"Spell the word: {word}"},
            {"role": "assistant", "content": f"{word}:{word_letters}"},
        ]
        return {"messages": messages}


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model")
# Logging / bookkeeping
parser.add_argument("--run-path", type=str, default="d32", help="Run name (dummy = disable external logging)")
parser.add_argument("--dry-run", action="store_true", help="Skip checkpoint write (for smoke tests)")
# Runtime
parser.add_argument("--device-type", type=str, default="", choices=["", "cuda", "cpu", "mps"], help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="Base checkpoint tag under weights/base_checkpoints/ (default: latest/largest)")
parser.add_argument("--model-step", type=int, default=None, help="Step to load (default: latest in the tag dir)")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="Number of optimizer steps (-1 = one epoch over train mix)")
# Batch sizes
parser.add_argument("--max-seq-len", type=int, default=2048, help="Max context length")
parser.add_argument("--device-batch-size", type=int, default=32, help="Per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=524288, help="Global tokens per step (B*T*world_size*grad_accum)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.3, help="Embedding LR (AdamW)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="Unembedding LR (AdamW)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="Matrix LR (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for AdamW params")
parser.add_argument("--init-lr-frac", type=float, default=1.0, help="Scale initial LR")
parser.add_argument("--grad-clip", type=float, default=0.0, help="Global grad clip (default 0 = disabled, to mirror chat_sft)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=150, help="Run val bpb every N steps (-1 to disable)")
parser.add_argument("--eval-tokens", type=int, default=20 * 524288, help="Tokens to eval on")
# Data
parser.add_argument("--identity-jsonl", type=str, default=None, help="Optional identity conversations jsonl")
parser.add_argument("--extra-jsonl", action="append", default=[], help="Additional custom jsonl files (can repeat)")
parser.add_argument("--hf-limit", type=int, default=None, help="Limit HF datasets for quick tests")
parser.add_argument("--jsonl-dir", type=str, default=None, help="Directory containing exported jsonl datasets (default: app/data/sft/jsonl)")

args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# Init compute / device / dtype
# -----------------------------------------------------------------------------

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else torch.no_grad()
synchronize = torch.cuda.synchronize if device_type == "cuda" else (lambda: None)
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else (lambda: 0)

print0(f"DDP world size: {ddp_world_size}")

# -----------------------------------------------------------------------------
# Load base model + tokenizer
# -----------------------------------------------------------------------------

base_dir = get_base_dir()
base_ckpt_root = os.path.join(base_dir, "weights", "base_checkpoints")
if args.model_tag is None:
    args.model_tag = find_largest_model(base_ckpt_root)
base_ckpt_dir = os.path.join(base_ckpt_root, args.model_tag)
if not os.path.isdir(base_ckpt_dir):
    raise FileNotFoundError(f"Base checkpoint directory not found: {base_ckpt_dir}")
base_step = args.model_step if args.model_step is not None else find_last_step(base_ckpt_dir)
print0(f"Loading base checkpoint tag={args.model_tag} step={base_step}")
model, tokenizer, meta = build_model(base_ckpt_dir, base_step, device, phase="train", allow_missing_c_gate=True)
model_config_kwargs = meta["model_config"] if "model_config" in meta else vars(model.config)
orig_model = model
model = torch.compile(model, dynamic=False)
num_flops_per_token = model.estimate_flops()
print0(f"FLOPs/token estimate: {num_flops_per_token:e}")

# Token bytes for bpb metric
token_bytes = get_token_bytes(device=device)

# -----------------------------------------------------------------------------
# Batch / grad accumulation
# -----------------------------------------------------------------------------

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0, "total_batch_size must be divisible by world_tokens_per_fwdbwd"
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd

print0(
    f"Tokens per micro-batch per rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}"
)
print0(f"Tokens per micro-batch (all ranks): {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => grad_accum_steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Optimizer setup
# -----------------------------------------------------------------------------

optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# Scale initial LR by init_lr_frac
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

# -----------------------------------------------------------------------------
# Data mixture (close to nanochat chat_sft)
# -----------------------------------------------------------------------------

def maybe_path(path: str | None) -> str | None:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


jsonl_root = args.jsonl_dir or os.path.join(get_sft_data_dir(), "jsonl")
jsonl_root = jsonl_root if os.path.isabs(jsonl_root) else os.path.join(base_dir, jsonl_root)

identity_path = maybe_path(args.identity_jsonl)
if identity_path is None:
    default_identity = os.path.join(jsonl_root, "identity_conversations.jsonl")
    fallback_identity = os.path.join(get_sft_data_dir(), "identity_conversations.jsonl")
    if os.path.exists(default_identity):
        identity_path = default_identity
    elif os.path.exists(fallback_identity):
        identity_path = fallback_identity

# Build train mixture from jsonl exports
def jsonl_path(filename: str):
    return os.path.join(jsonl_root, filename)

train_tasks: List[Task] = [
    CustomJSON(jsonl_path("smol_smoltalk_train.jsonl")),
    CustomJSON(jsonl_path("mmlu_auxiliary_train_train.jsonl")),
    CustomJSON(jsonl_path("gsm8k_main_train.jsonl")),
    CustomJSON(jsonl_path("gsm8k_main_train.jsonl")),  # oversample GSM8K 2x
]
if identity_path is not None and os.path.exists(identity_path):
    train_tasks.append(CustomJSON(identity_path))
    train_tasks.append(CustomJSON(identity_path))  # 2 epochs of identity
for extra in args.extra_jsonl:
    extra_path = maybe_path(extra)
    if extra_path is not None:
        train_tasks.append(CustomJSON(extra_path))

# Add spelling-focused tasks after identity data to mirror nanochat mix order
train_tasks.extend([
    SimpleSpelling(size=200_000, split="train"),
    SpellingBee(size=80_000, split="train"),
])

train_dataset = TaskMixture(train_tasks, seed=42)

val_tasks: List[Task] = [
    CustomJSON(jsonl_path("smol_smoltalk_test.jsonl")),
    CustomJSON(jsonl_path("mmlu_all_test.jsonl"), stop=5200),
    CustomJSON(jsonl_path("gsm8k_main_test.jsonl"), stop=420),
]
val_dataset = TaskMixture(val_tasks, seed=7)

print0(
    f"SFT train mix: {len(train_dataset):,} convs | val mix: {len(val_dataset):,} convs"
)

# -----------------------------------------------------------------------------
# SFT DataLoader with BOS-aligned best-fit packing
# -----------------------------------------------------------------------------


def sft_data_generator(split: str, state: Dict[str, Any], buffer_size: int = 100):
    """
    Packs conversations into fixed-length rows using best-fit (no token discarded).
    Returns (inputs, targets) tensors on CPU (pinned if CUDA) ready for GPU transfer.
    """

    assert split in {"train", "val"}
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    row_capacity = args.max_seq_len + 1  # +1 for target shift
    bos_token = tokenizer.get_bos_token_id()

    conv_buffer: List[Tuple[List[int], List[int]]] = []
    cursor = ddp_rank  # stagger across ranks
    consumed = ddp_rank
    epoch = 1
    it = 0

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conv = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conv, max_tokens=args.max_seq_len)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1

    while True:
        rows: List[List[int]] = []
        masks: List[List[int]] = []
        row_lengths: List[int] = []
        for _ in range(args.device_batch_size):
            row_ids: List[int] = []
            row_mask: List[int] = []
            padded = False
            while len(row_ids) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - len(row_ids)
                # pick largest conversation that fits
                best_idx = -1
                best_len = -1
                for i, (ids, _) in enumerate(conv_buffer):
                    if len(ids) <= remaining and len(ids) > best_len:
                        best_idx = i
                        best_len = len(ids)
                if best_idx >= 0:
                    ids, mask = conv_buffer.pop(best_idx)
                    row_ids.extend(ids)
                    row_mask.extend(mask)
                    consumed += ddp_world_size
                else:
                    content_len = len(row_ids)
                    pad_len = remaining
                    row_ids.extend([bos_token] * pad_len)
                    row_mask.extend([-1] * pad_len)
                    padded = True
                    break
            row_lengths.append(content_len if padded else row_capacity)
            rows.append(row_ids[:row_capacity])
            masks.append(row_mask[:row_capacity])

        # Update progress info (train only)
        it += 1
        if split == "train":
            state["current_epoch"] = epoch
            if args.num_iterations > 0:
                state["approx_progress"] = min(it / args.num_iterations, 1.0)
            else:
                state["approx_progress"] = min(consumed / max(dataset_size, 1), 1.0)
            if consumed >= dataset_size:
                state["last_step"] = True

        # Build tensors
        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        mask_tensor = torch.tensor(masks, dtype=torch.int32, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)
        target_mask = mask_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)
        # mask: supervise only where mask > 0, ignore others (0 or -1)
        targets = torch.where(target_mask > 0, targets, torch.full_like(targets, -1))
        yield inputs, targets


train_state = {"last_step": False, "approx_progress": 0.0, "current_epoch": 1}
train_loader = sft_data_generator("train", state=train_state)
build_val_loader = lambda: sft_data_generator("val", state={"last_step": False, "approx_progress": 0.0, "current_epoch": 1})
x, y = next(train_loader)  # prefetch first batch

# -----------------------------------------------------------------------------
# LR & momentum schedule (match nanochat chat_sft)
# -----------------------------------------------------------------------------


def get_lr_multiplier(progress: float) -> float:
    return 1.0 if progress < 0.8 else max(0.0, 1 - (progress - 0.8) / 0.2)


def get_muon_momentum(it: int) -> float:
    frac = min(it / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


# -----------------------------------------------------------------------------
# Checkpoint dir
# -----------------------------------------------------------------------------

ckpt_root = os.path.join(base_dir, "weights", "sft")
output_dirname = args.run_path
checkpoint_dir = os.path.join(ckpt_root, output_dirname)
if ddp_rank == 0:
    os.makedirs(checkpoint_dir, exist_ok=True)
if ddp:
    dist.barrier()

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

step = 0
min_val_bpb = float("inf")
smooth_train_loss = 0.0
total_training_time = 0.0

max_steps = args.num_iterations if args.num_iterations > 0 else None

while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # -------- Eval --------
    if args.eval_every != -1 and step % args.eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size))
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        min_val_bpb = min(min_val_bpb, val_bpb)
        print0(f"[Eval] step {step:05d} | val bpb: {val_bpb:.4f} | min: {min_val_bpb:.4f}")
        model.train()

    # -------- Train step --------
    synchronize()
    t0 = time.time()

    for micro in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)

    if args.grad_clip > 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), args.grad_clip)
        grad_norm = grad_norm_tensor.item()
    else:
        grad_norm = None

    # LR and Muon momentum schedule
    progress = train_state["approx_progress"]
    lrm = get_lr_multiplier(progress)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    if len(optimizers) > 1:
        for group in optimizers[1].param_groups:
            group["momentum"] = muon_momentum

    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    synchronize()
    dt = time.time() - t0
    if step > 10:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))

    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100
    pct_done = progress * 100 if max_steps is None else 100 * step / max_steps

    grad_str = f" | grad_norm: {grad_norm:.4f}" if grad_norm is not None else ""
    print0(
        f"step {step:05d}"
        f" | loss: {debiased_loss:.6f}"
        f"{grad_str}"
        f" | lrm: {lrm:.2f}"
        f" | muon_m: {muon_momentum:.3f}"
        f" | dt: {dt * 1000:.1f}ms"
        f" | tok/sec: {tok_per_sec:,}"
        f" | mfu: {mfu:.2f}"
        f" | progress: {pct_done:5.2f}%"
    )

    step += 1
    done_by_steps = max_steps is not None and step >= max_steps
    done_by_epoch = max_steps is None and train_state.get("last_step", False)
    if done_by_steps or done_by_epoch:
        break

# -----------------------------------------------------------------------------
# Final eval & checkpoint
# -----------------------------------------------------------------------------

model.eval()
val_loader = build_val_loader()
eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size))
with autocast_ctx:
    val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
min_val_bpb = min(min_val_bpb, val_bpb)
print0(f"[Final Eval] step {step:05d} | val bpb: {val_bpb:.4f} | min: {min_val_bpb:.4f}")

meta_data = {
    "step": step,
    "val_bpb": float(val_bpb),
    "min_val_bpb": float(min_val_bpb),
    "model_config": model_config_kwargs,
    "user_config": user_config,
    "base_checkpoint": {"tag": args.model_tag, "step": base_step},
    "loop_state": {
        "smooth_train_loss": float(smooth_train_loss),
        "total_training_time": float(total_training_time),
    },
}

if not args.dry_run:
    save_checkpoint(
        checkpoint_dir,
        step,
        orig_model.state_dict(),
        [opt.state_dict() for opt in optimizers],
        meta_data,
        rank=ddp_rank,
    )
    print0(f"Checkpoint saved to {checkpoint_dir}")
else:
    print0("Dry run enabled; checkpoint not written.")

print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f} MiB")
compute_cleanup()
