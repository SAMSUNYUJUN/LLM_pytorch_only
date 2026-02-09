"""
Continue supervised fine-tuning (SFT) from an existing run, focusing on the
identity_conversations.jsonl data (duplicated 2x).

示例：
1) 默认加载最新 checkpoint 并继续训练一轮（跑满两份 identity 数据）：
   CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
   app/modules/sft/continue_sft.py \
   --device-type cuda \
   --run-path d32 \
   --lrm 0.1

2) 单卡冒烟 40 步：
   python app/modules/sft/continue_sft.py --device-type cuda --device-batch-size 2 --total-batch-size 8192 --num-iterations 40 --lrm 0.05

特性：
- 总是从指定 run-path（weights/sft/<run-path>）的最新 checkpoint 继续训练；也可通过 --resume-step 指定。
- 训练数据仅包含两份 identity_conversations.jsonl，实现 2x upsample。
- 可通过 --lrm 指定学习率乘子，覆盖进度调度；训练结束前会将 lr 恢复为加载 checkpoint 时的值，避免破坏原始优化器状态。
"""

import os
import sys
import json
import time
import argparse
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
    ) 
    from ..utils.checkpoint_manager import (
        build_model,
        find_last_step,
        load_checkpoint,
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
    )
    from app.modules.utils.checkpoint_manager import (
        build_model,
        find_last_step,
        load_checkpoint,
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
# Argument parsing
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Continue SFT from latest checkpoint on identity data (2x)")
# Logging / bookkeeping
parser.add_argument("--run-path", type=str, default="d32", help="Run name under weights/sft/ to resume from")
parser.add_argument("--dry-run", action="store_true", help="Skip checkpoint write (for smoke tests)")
parser.add_argument("--resume-step", type=int, default=None, help="Resume from a specific checkpoint step (default: latest)")
# Runtime
parser.add_argument("--device-type", type=str, default="", choices=["", "cuda", "cpu", "mps"], help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="Number of optimizer steps (-1 = one epoch over 2x identity)")
parser.add_argument("--save-every", type=int, default=400, help="Checkpoint every N steps (model + optimizer)")
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
parser.add_argument("--grad-clip", type=float, default=0.0, help="Global grad clip (default 0 = disabled)")
parser.add_argument("--lrm", type=float, default=None, help="Override lr multiplier during continue phase (constant). None = use schedule.")
parser.add_argument("--no-restore-lrm", action="store_true", help="Do not restore original lr multipliers before saving final checkpoint")
# Evaluation
parser.add_argument("--eval-every", type=int, default=150, help="Run val bpb every N steps (-1 to disable)")
parser.add_argument("--eval-tokens", type=int, default=20 * 524288, help="Tokens to eval on")
# Data
parser.add_argument("--identity-jsonl", type=str, default=None, help="Optional identity conversations jsonl path")
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
# Load latest SFT checkpoint + tokenizer
# -----------------------------------------------------------------------------

base_dir = get_base_dir()
sft_ckpt_root = os.path.join(base_dir, "weights", "sft")
checkpoint_dir = os.path.join(sft_ckpt_root, args.run_path)
if not os.path.isdir(checkpoint_dir):
    raise FileNotFoundError(f"SFT checkpoint directory not found: {checkpoint_dir}")

resume_step = args.resume_step if args.resume_step is not None else find_last_step(checkpoint_dir)
print0(f"Resuming from {checkpoint_dir} step={resume_step}")

model, tokenizer, meta = build_model(checkpoint_dir, resume_step, device, phase="train", allow_missing_c_gate=True)
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
# Optimizer setup (load from checkpoint, allow temporary LR override)
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

# Load optimizer state for this rank if available
optim_path = os.path.join(checkpoint_dir, f"optim_{resume_step:06d}_rank{ddp_rank:d}.pt")
if os.path.exists(optim_path):
    optimizer_state = torch.load(optim_path, map_location=device)
    if isinstance(optimizer_state, list):
        for opt, state in zip(optimizers, optimizer_state):
            opt.load_state_dict(state)
    else:
        # Backward compatibility: single-optimizer checkpoint
        for opt in optimizers:
            opt.load_state_dict(optimizer_state)
else:
    print0(f"Optimizer state not found at {optim_path}, starting with fresh optimizer state")


def capture_lrm(optim_list):
    captured: List[List[float]] = []
    for opt in optim_list:
        opt_lrm: List[float] = []
        for group in opt.param_groups:
            initial_lr = group.get("initial_lr", group["lr"])
            if initial_lr == 0:
                opt_lrm.append(0.0)
            else:
                opt_lrm.append(group["lr"] / initial_lr)
        captured.append(opt_lrm)
    return captured


loaded_lrms = capture_lrm(optimizers)

# -----------------------------------------------------------------------------
# Data mixture: identity only (2x)
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
    else:
        raise FileNotFoundError("identity_conversations.jsonl not found in provided or default locations")


def jsonl_path(filename: str):
    return os.path.join(jsonl_root, filename)


train_tasks: List[Task] = [
    CustomJSON(identity_path),
    CustomJSON(identity_path),  # 2x upsample
]

train_dataset = TaskMixture(train_tasks, seed=42)

val_tasks: List[Task] = [CustomJSON(identity_path)]
val_dataset = TaskMixture(val_tasks, seed=7)

print0(f"Continue SFT train mix: {len(train_dataset):,} convs | val mix: {len(val_dataset):,} convs")

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
    cursor = state.get("cursor", ddp_rank)  # stagger across ranks
    consumed = state.get("consumed", ddp_rank)
    epoch = state.get("epoch", 1)
    it = state.get("it", 0)


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
        state["cursor"] = cursor
        state["consumed"] = consumed
        state["epoch"] = epoch
        state["it"] = it
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


# -----------------------------------------------------------------------------
# LR & momentum schedule (match nanochat chat_sft)
# -----------------------------------------------------------------------------


def get_lr_multiplier(progress: float) -> float:
    return 1.0 if progress < 0.8 else max(0.0, 1 - (progress - 0.8) / 0.2)


def get_muon_momentum(it: int) -> float:
    frac = min(it / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


def resolve_lrm(progress: float) -> float:
    if args.lrm is not None:
        return args.lrm
    return get_lr_multiplier(progress)


# -----------------------------------------------------------------------------
# Resume logic and dataloaders
# -----------------------------------------------------------------------------

loaded_train_state: Dict[str, Any] = meta.get("train_state", {}) or {}
last_val_bpb = meta.get("val_bpb", None)
min_val_bpb = meta.get("min_val_bpb", float("inf"))
loop_state = meta.get("loop_state", {}) or {}
smooth_train_loss = loop_state.get("smooth_train_loss", 0.0)
total_training_time = loop_state.get("total_training_time", 0.0)
step = meta.get("step", resume_step)

# For the continue phase we reset data cursors to iterate over the new 2x-identity mix
train_state = {
    "last_step": False,
    "approx_progress": 0.0,
    "current_epoch": 1,
    "cursor": ddp_rank,
    "consumed": ddp_rank,
    "epoch": 1,
    "it": 0,
}

train_loader = sft_data_generator("train", state=train_state)
build_val_loader = lambda: sft_data_generator(
    "val", state={"last_step": False, "approx_progress": 0.0, "current_epoch": 1}
)
x, y = next(train_loader)  # prefetch first batch


def build_meta_data(step_value: int, val_bpb_value):
    train_state_to_save = {k: v for k, v in train_state.items()}
    # Avoid carrying termination flag into next run
    train_state_to_save["last_step"] = False
    return {
        "step": step_value,
        "val_bpb": None if val_bpb_value is None else float(val_bpb_value),
        "min_val_bpb": float(min_val_bpb),
        "model_config": model_config_kwargs,
        "user_config": user_config,
        "base_checkpoint": meta.get("base_checkpoint", {"tag": args.run_path, "step": resume_step}),
        "loop_state": {
            "smooth_train_loss": float(smooth_train_loss),
            "total_training_time": float(total_training_time),
        },
        "train_state": train_state_to_save,
    }


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

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
        last_val_bpb = val_bpb
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
    lrm = resolve_lrm(progress)
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

    # Periodic checkpointing (after increment so step matches completed steps)
    if (not args.dry_run) and args.save_every > 0 and step % args.save_every == 0:
        meta_data = build_meta_data(step, last_val_bpb)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            meta_data,
            rank=ddp_rank,
        )
        print0(f"Checkpoint saved to {checkpoint_dir} at step {step:05d}")


# -----------------------------------------------------------------------------
# Final eval & checkpoint (restore original lrm before saving)
# -----------------------------------------------------------------------------

model.eval()
val_loader = build_val_loader()
eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size))
with autocast_ctx:
    val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
min_val_bpb = min(min_val_bpb, val_bpb)
last_val_bpb = val_bpb
print0(f"[Final Eval] step {step:05d} | val bpb: {val_bpb:.4f} | min: {min_val_bpb:.4f}")

# Restore original lr multipliers if requested
if not args.no_restore_lrm:
    for opt, lrms in zip(optimizers, loaded_lrms):
        for group, lrm_value in zip(opt.param_groups, lrms):
            group["lr"] = group.get("initial_lr", group["lr"]) * lrm_value

meta_data = build_meta_data(step, val_bpb)

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
