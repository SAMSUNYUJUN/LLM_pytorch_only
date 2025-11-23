"""
Usage (single GPU):
    python pretrain/pretrain.py

Usage (multi-GPU):
    torchrun --nproc_per_node=2 pretrain/pretrain.py
"""

import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 当前文件的绝对路径：.../LLM_pytorch_only/pretrain/pretrain.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录：.../LLM_pytorch_only
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 把项目根目录加到 sys.path 里
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
from contextlib import nullcontext

import torch


from model.gpt import GPT, GPTConfig
from model.dataloader import (
    tokenizing_distributed_data_loader,
    tokenizing_distributed_data_loader_with_state,
)
from utils.utils import (
    compute_init,
    compute_cleanup,
    print0,
    get_base_dir,
    autodetect_device_type,
)
from tokenizer.tokenizer import get_tokenizer, get_token_bytes
from utils.checkpoint_manager import save_checkpoint
from model.loss_eval import evaluate_bpb

# ==========================
# 固定超参（可按需自己改）
# ==========================

DEPTH = 2               # Transformer 层数
MAX_SEQ_LEN = 2048       # 上下文长度
DEVICE_BATCH_SIZE = 16   # 单卡 batch（按内存调）
TOTAL_BATCH_SIZE = 524288  # 总 token batch size
TARGET_PARAM_DATA_RATIO = 20  # Chinchilla 比例，tokens/params ~= 20

EVAL_EVERY = 250         # 每多少 step 跑一次 val bpb
EVAL_TOKENS = 20 * 524288  # val 上的 token 数量
GRAD_CLIP = 1.0
EMBEDDING_LR = 0.2
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.02
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.2
FINAL_LR_FRAC = 0.0

MODEL_TAG = ""  # 输出目录名（默认 d20）


# ==========================
# 初始化分布式 / 设备
# ==========================

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # 只在 rank0 打 log / 保存
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda"
    else nullcontext()
)
synchronize = torch.cuda.synchronize if device_type == "cuda" else (lambda: None)
get_max_memory = (
    torch.cuda.max_memory_allocated if device_type == "cuda" else (lambda: 0)
)

# ==========================
# Tokenizer & vocab
# ==========================

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# ==========================
# 模型结构超参（跟原 base_train 一致）
# ==========================

num_layers = DEPTH
model_dim = DEPTH * 64  # aspect ratio 64
num_heads = max(1, (model_dim + 127) // 128)  # head dim 128
num_kv_heads = num_heads  # 关闭 GQA（1:1）

print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# ==========================
# batch / grad accumulation
# ==========================

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert TOTAL_BATCH_SIZE % world_tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // world_tokens_per_fwdbwd

print0(
    f"Tokens / micro-batch / rank: {DEVICE_BATCH_SIZE} x {MAX_SEQ_LEN} = {tokens_per_fwdbwd:,}"
)
print0(f"Tokens / micro-batch (all ranks): {world_tokens_per_fwdbwd:,}")
print0(
    f"Total batch size {TOTAL_BATCH_SIZE:,} => gradient accumulation steps: {grad_accum_steps}"
)

# ==========================
# 构建模型
# ==========================

model_config_kwargs = dict(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
)

with torch.device("meta"):
    gpt_config = GPTConfig(**model_config_kwargs)
    model = GPT(gpt_config)

model.to_empty(device=device)
model.init_weights()

orig_model = model  # 保存未 compile 的版本，用来存权重
model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# ==========================
# 计算训练步数（用 data:param ratio）
# ==========================

target_tokens = TARGET_PARAM_DATA_RATIO * num_params
num_iterations = target_tokens // TOTAL_BATCH_SIZE
assert num_iterations > 0
total_tokens = TOTAL_BATCH_SIZE * num_iterations

print0(f"Calculated number of iterations: {num_iterations:,}")
print0(f"Total training tokens: {total_tokens:,}")
print0(
    f"Tokens : Params ratio: {TOTAL_BATCH_SIZE * num_iterations / num_params:.2f}"
)
print0(
    f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}"
)

# ==========================
# Optimizer
# ==========================

optimizers = model.setup_optimizers(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)
adamw_optimizer, muon_optimizer = optimizers

# ==========================
# DataLoader
# ==========================

base_dir = get_base_dir()
tokens_dir = os.path.join(base_dir, "tokenized_data")

train_loader = tokenizing_distributed_data_loader_with_state(
    DEVICE_BATCH_SIZE,
    MAX_SEQ_LEN,
    split="train",
    device=device,
    resume_state_dict=None,
)
build_val_loader = lambda: tokenizing_distributed_data_loader(
    DEVICE_BATCH_SIZE, MAX_SEQ_LEN, split="val", device=device
)

x, y, dataloader_state_dict = next(train_loader)

# ==========================
# LR & Muon momentum schedule
# ==========================


def get_lr_multiplier(it: int) -> float:
    warmup_iters = round(WARMUP_RATIO * num_iterations)
    warmdown_iters = round(WARMDOWN_RATIO * num_iterations)
    if warmup_iters > 0 and it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / max(warmdown_iters, 1)
        return progress * 1.0 + (1 - progress) * FINAL_LR_FRAC


def get_muon_momentum(it: int) -> float:
    frac = min(it / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


# ==========================
# 训练循环
# ==========================

step = 0
min_val_bpb = float("inf")
smooth_train_loss = 0.0
total_training_time = 0.0

# 准备 checkpoint 目录
output_dirname = MODEL_TAG if MODEL_TAG else f"d{DEPTH}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
if master_process:
    os.makedirs(checkpoint_dir, exist_ok=True)
if ddp:
    torch.distributed.barrier()

while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * TOTAL_BATCH_SIZE * step

    # ------- eval on val set -------
    if last_step or step % EVAL_EVERY == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = EVAL_TOKENS // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"[Eval] Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        min_val_bpb = min(min_val_bpb, val_bpb)
        model.train()

    # ------- 保存 checkpoint（仅最后一步 & 只在 master） -------
    if last_step:
        if master_process:
            meta = {
                "step": step,
                "val_bpb": float(val_bpb),
                "model_config": model_config_kwargs,
                "user_config": {
                    "depth": DEPTH,
                    "max_seq_len": MAX_SEQ_LEN,
                    "device_batch_size": DEVICE_BATCH_SIZE,
                    "total_batch_size": TOTAL_BATCH_SIZE,
                    "target_param_data_ratio": TARGET_PARAM_DATA_RATIO,
                },
                "device_batch_size": DEVICE_BATCH_SIZE,
                "max_seq_len": MAX_SEQ_LEN,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {
                    "min_val_bpb": float(min_val_bpb),
                    "smooth_train_loss": float(smooth_train_loss),
                    "total_training_time": float(total_training_time),
                },
            }
            # 只保存模型权重 + meta，不保存 optimizer 状态
            save_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict(),
                [],  # no optimizer states
                meta,
                rank=0,
                model_name="model_final.pt",
                meta_name="meta_final.json",
            )
            print0(f"Saved final weights to {checkpoint_dir}/model_final.pt")
        break

    # ------- 单步训练 -------
    synchronize()
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)

    # gradient clipping
    if GRAD_CLIP > 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            orig_model.parameters(), GRAD_CLIP
        )
        grad_norm = grad_norm_tensor.item()
    else:
        grad_norm = None

    # update LR & Muon momentum
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum

    # optimizer step
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # logging（只打印到控制台）
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    flops_per_sec = num_flops_per_token * TOTAL_BATCH_SIZE / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100

    if step > 10:
        total_training_time += dt

    grad_str = f" | grad_norm: {grad_norm:.4f}" if grad_norm is not None else ""
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:5.2f}%)"
        f" | loss: {debiased_smooth_loss:.6f}"
        f"{grad_str}"
        f" | lrm: {lrm:.2f}"
        f" | dt: {dt * 1000:.2f}ms"
        f" | tok/sec: {tok_per_sec:,}"
        f" | mfu: {mfu:.2f}"
        f" | total time: {total_training_time/60:.2f}m"
    )

    step += 1

# ==========================
# 训练结束总结
# ==========================

print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

compute_cleanup()

