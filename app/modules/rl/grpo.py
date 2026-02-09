"""
GRPO-style reinforcement learning on GSM8K (local jsonl exports).

Usage examples:

1) 单卡冒烟：
   python -m app.modules.rl.grpo \
     --device-type cuda \
     --device-batch-size 2 --examples-per-step 4 --num-samples 4 \
     --eval-every 10 --save-every 10 --num-epochs 1 --run-path d32_smoke

2) 多卡（4 卡）：
   torchrun --standalone --nproc_per_node=4 -m app.modules.rl.grpo --run-path d32

默认从 weights/sft 下选择最大的 checkpoint 作为初始策略，
输出保存到 weights/rl/<run-path>/model_*.pt（默认 run-path=d32）。
"""

import argparse
import itertools
import json
import os
import sys
from contextlib import nullcontext

import torch
import torch.distributed as dist

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ----------------------------------------------------------------------------
# Imports with fallback for direct invocation
# ----------------------------------------------------------------------------
try:
    from ..utils.utils import compute_init, compute_cleanup, autodetect_device_type, get_base_dir, print0
    from ..utils.checkpoint_manager import (
        build_model,
        find_last_step,
        find_largest_model,
        save_checkpoint,
    )
    from ..model.engine import Engine
except ImportError:  # pragma: no cover - allow running as a script
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from app.modules.utils.utils import compute_init, compute_cleanup, autodetect_device_type, get_base_dir, print0
    from app.modules.utils.checkpoint_manager import (
        build_model,
        find_last_step,
        find_largest_model,
        save_checkpoint,
    )
    from app.modules.model.engine import Engine


# ----------------------------------------------------------------------------
# GSM8K dataset (jsonl) + reward helpers
# ----------------------------------------------------------------------------

import re

ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(text: str) -> str | None:
    """Extract numeric answer following the GSM8K "####" marker."""

    match = ANSWER_RE.search(text)
    if match:
        out = match.group(1).strip().replace(",", "")
        return out
    return None


class Task:
    def __len__(self):
        return self.num_examples()

    def num_examples(self):
        raise NotImplementedError

    def __getitem__(self, idx: int):
        return self.get_example(idx)

    def get_example(self, idx: int):  # pragma: no cover - interface only
        raise NotImplementedError


class GSM8KJsonl(Task):
    """Lightweight GSM8K loader based on exported jsonl conversations."""

    def __init__(self, split: str, jsonl_root: str | None = None, limit: int | None = None):
        assert split in {"train", "test"}, "split must be train|test"
        base_dir = get_base_dir()
        root = jsonl_root or os.path.join(base_dir, "data", "sft", "jsonl")
        filename = f"gsm8k_main_{split}.jsonl"
        if os.path.isdir(root):
            path = os.path.join(root, filename)
        else:
            path = root  # allow passing the file directly
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        assert os.path.exists(path), f"GSM8K jsonl not found: {path}"
        self.path = path
        self.data = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))
                if limit is not None and len(self.data) >= limit:
                    break

    def num_examples(self):
        return len(self.data)

    def get_example(self, idx: int):
        # Keep consistent with sft.py: return {"messages": [...]}
        return self.data[idx]

    def _reference_answer(self, conversation) -> str | None:
        messages = conversation["messages"] if isinstance(conversation, dict) else conversation
        assert messages[-1]["role"] == "assistant", "Last message must be assistant"
        content = messages[-1]["content"]
        if isinstance(content, str):
            joined = content
        elif isinstance(content, list):
            joined = "".join(part.get("text", "") for part in content)
        else:
            joined = str(content)
        return extract_answer(joined)

    def evaluate(self, conversation, assistant_response: str) -> int:
        ref = self._reference_answer(conversation)
        pred = extract_answer(assistant_response)
        return int(ref is not None and pred == ref)

    def reward(self, conversation, assistant_response: str) -> float:
        return float(self.evaluate(conversation, assistant_response))


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="GRPO on GSM8K (jsonl)")
# Logging / bookkeeping (match sft.py naming)
parser.add_argument("--run-path", type=str, default="d32", help="Checkpoint subdir under weights/rl (default: d32)")
# Runtime
parser.add_argument("--device-type", type=str, default="", choices=["", "cuda", "cpu", "mps"], help="Device type (empty=autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"], help="Training dtype")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="SFT checkpoint tag under weights/sft (default: largest)")
parser.add_argument("--model-step", type=int, default=None, help="Step to load (default: last step in tag dir)")
# Data
parser.add_argument("--jsonl-dir", type=str, default=None, help="Directory containing gsm8k_main_{split}.jsonl (default: app/data/sft/jsonl)")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs over GSM8K train set")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="Max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="Examples per optimization step (global across ranks)")
parser.add_argument("--num-samples", type=int, default=16, help="Samples per example/question")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (0 disables)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="LR for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="LR for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="LR for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for Adam params")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="Initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="Evaluate pass@k every N steps (-1 disables)")
parser.add_argument("--eval-examples", type=int, default=400, help="Number of examples for evaluation")
parser.add_argument("--save-every", type=int, default=60, help="Checkpoint every N steps (rank0)")

args = parser.parse_args()
user_config = vars(args).copy()

# ----------------------------------------------------------------------------
# Init compute/precision
# ----------------------------------------------------------------------------

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# Init model + tokenizer
# ----------------------------------------------------------------------------

base_dir = get_base_dir()
sft_root = os.path.join(base_dir, "weights", "sft")
if args.model_tag is None:
    args.model_tag = find_largest_model(sft_root)
ckpt_dir = os.path.join(sft_root, args.model_tag)
if not os.path.isdir(ckpt_dir):
    raise FileNotFoundError(f"SFT checkpoint directory not found: {ckpt_dir}")
ckpt_step = args.model_step if args.model_step is not None else find_last_step(ckpt_dir)
print0(f"Loading SFT checkpoint tag={args.model_tag} step={ckpt_step}")
model, tokenizer, meta = build_model(ckpt_dir, ckpt_step, device, phase="train", allow_missing_c_gate=True)
engine = Engine(model, tokenizer)

# ----------------------------------------------------------------------------
# Dataset setup
# ----------------------------------------------------------------------------

train_task = GSM8KJsonl(split="train", jsonl_root=args.jsonl_dir)
val_task = GSM8KJsonl(split="test", jsonl_root=args.jsonl_dir)
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
if num_steps == 0:
    raise ValueError("num_steps computed to 0; check examples_per_step / num_epochs")
print0(f"Calculated number of steps: {num_steps}")


@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    assert args.num_samples % args.device_batch_size == 0, "num_samples must be divisible by device_batch_size"
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
    for example_idx in itertools.cycle(rank_indices):
        conversation = train_task[example_idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        model.eval()
        generated_token_sequences = []
        masks = []
        num_sampling_steps = args.num_samples // args.device_batch_size
        for sampling_step in range(num_sampling_steps):
            seed_val = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF  # step defined in training loop
            with autocast_ctx:
                gen_batch, mask_batch = engine.generate_batch(
                    tokens,
                    num_samples=args.device_batch_size,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=None if args.top_k == 0 else args.top_k,
                    seed=seed_val,
                )
            generated_token_sequences.extend(gen_batch)
            masks.extend(mask_batch)

        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            rewards.append(train_task.reward(conversation, generated_text))

        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]

        ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1

        rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)
        advantages = rewards_t - rewards_t.mean()
        yield generated_token_sequences, inputs, targets, rewards_t, advantages


def run_gsm8k_eval(task: GSM8KJsonl, tokenizer, engine: Engine,
                   max_examples=None, num_samples=1, max_completion_tokens=256, temperature=0.0, top_k=50):
    """Yield evaluation records (non-reduced across DDP ranks)."""

    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        assert num_samples <= args.device_batch_size
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})
        yield {"idx": idx, "outcomes": outcomes}


# ----------------------------------------------------------------------------
# Optimizer setup
# ----------------------------------------------------------------------------

optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]


def get_lr_multiplier(it):
    return 1.0 - it / num_steps


print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}")
assert args.examples_per_step % ddp_world_size == 0, "examples_per_step must be divisible by world size"
examples_per_rank = args.examples_per_step // ddp_world_size
print0(f"Calculated examples per rank: {examples_per_rank}")

# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------

batch_iterator = get_batch()
checkpoint_root = os.path.join(base_dir, "weights", "rl")
if master_process:
    os.makedirs(os.path.join(checkpoint_root, args.run_path), exist_ok=True)
if ddp:
    dist.barrier()

for step in range(num_steps):

    if args.eval_every != -1 and step % args.eval_every == 0:
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device)
        with autocast_ctx:
            records = list(
                run_gsm8k_eval(
                    val_task,
                    tokenizer,
                    engine,
                    num_samples=args.device_batch_size,
                    max_examples=args.eval_examples,
                    temperature=1.0,
                    top_k=None if args.top_k == 0 else args.top_k,
                )
            )
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / max(num_records.item(), 1)
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")

    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        model.train()
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            with autocast_ctx:
                logp = -model(inputs, targets, loss_reduction="none").view_as(inputs)
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            loss = -pg_obj
            loss.backward()
            print0(
                f"Step {step}/{num_steps} | Example {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Avg reward: {rewards.mean().item()}"
            )
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_seq_len = sum(sequence_lengths) / len(sequence_lengths)
    if ddp:
        reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        seq_tensor = torch.tensor(mean_seq_len, dtype=torch.float, device=device)
        dist.all_reduce(reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(seq_tensor, op=dist.ReduceOp.AVG)
        mean_reward = reward_tensor.item()
        mean_seq_len = seq_tensor.item()
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward:.4f} | Avg seq len: {mean_seq_len:.2f}")

    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        opt.step()
    model.zero_grad(set_to_none=True)

    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        ckpt_dir = os.path.join(checkpoint_root, args.run_path)
        meta_out = {
            "model_config": model.config.__dict__,
            "user_config": user_config,
            "sft_checkpoint": {"tag": args.model_tag, "step": ckpt_step},
        }
        save_checkpoint(ckpt_dir, step, model.state_dict(), None, meta_out, rank=ddp_rank)
        print0(f"✅ Saved model checkpoint to {ckpt_dir}")

compute_cleanup()
