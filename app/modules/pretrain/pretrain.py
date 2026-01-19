"""
torchrun --nproc_per_node=4 app/modules/pretrain/pretrain.py
"""

import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import torch
import json
import torch.distributed as dist

try:
    from ..model.gpt import GPT, GPTConfig
    from ..model.dataloader import (
        tokenizing_distributed_data_loader,
        tokenizing_distributed_data_loader_with_state,
    )
    from ..utils.utils import (
        get_base_dir,
    )
    from ..tokenizer.tokenizer import get_tokenizer, get_token_bytes
    from ..model.loss_eval import evaluate_bpb
except ImportError:
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from app.modules.model.gpt import GPT, GPTConfig
    from app.modules.model.dataloader import (
        tokenizing_distributed_data_loader,
        tokenizing_distributed_data_loader_with_state,
    )
    from app.modules.utils.utils import (
        get_base_dir,
    )
    from app.modules.tokenizer.tokenizer import get_tokenizer, get_token_bytes
    from app.modules.model.loss_eval import evaluate_bpb

def rank0_print(*args, **kwargs):
    # dist 还没 init 或单卡时，也允许正常打印
    if (not dist.is_available()) or (not dist.is_initialized()):
        print(*args, **kwargs)
        return

    if dist.get_rank() == 0:
        print(*args, **kwargs)

# ==========================
# 固定超参（可按需自己改）
# ==========================

DEPTH = 32               # Transformer 层数
MAX_SEQ_LEN = 2048       # 上下文长度
DEVICE_BATCH_SIZE = 16   # 单卡 batch（按内存调）
TOTAL_BATCH_SIZE = 524288  # 每个 step（做一次 optimizer.step 前）全局累计看到的 token 总数
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

device_type = "cuda"

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# 如果是用 torchrun 启动的，则会有RANK环境变量，如果没用torch run启动，则默认为单卡，环境变量被set为 -1
assert all(
        v in os.environ for v in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        ), "DDP mode requires RANK, LOCAL_RANK, WORLD_SIZE in env"

dist.init_process_group(backend="nccl", init_method="env://")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{rank}")  # Ensure each rank uses a different GPU
torch.cuda.set_device(device)
# 在 with 里面自动用 混合精度（bfloat16） 跑算子，省显存、提速。
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
# 把当前设备上的所有异步 CUDA 操作都等它跑完，再往下执行。
synchronize = torch.cuda.synchronize 
# 获取当前设备的最大显存使用量（字节）
get_max_memory = torch.cuda.max_memory_allocated 

# ==========================
# Tokenizer & vocab
# ==========================

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
rank0_print(f"Vocab size: {vocab_size:,}")

# ==========================
# 模型结构超参（跟原 base_train 一致）
# ==========================

num_layers = DEPTH
model_dim = DEPTH * 64  # aspect ratio 64
num_heads = max(1, (model_dim + 127) // 128)  # head dim 128
num_kv_heads = num_heads  # 关闭 GQA（1:1）

rank0_print(f"num_layers: {num_layers}")
rank0_print(f"model_dim: {model_dim}")
rank0_print(f"num_heads: {num_heads}")
rank0_print(f"num_kv_heads: {num_kv_heads}")

# ==========================
# batch / grad accumulation
# ==========================

# 每次forward/backward的总token数 = 单卡batch size x 上下文长度
tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
# all rank forward/backwork token 数 = 每次 forward/backward 的 token 数 x 卡数
world_tokens_per_fwdbwd = tokens_per_fwdbwd * world_size
assert TOTAL_BATCH_SIZE % world_tokens_per_fwdbwd == 0
# 梯度累积步数（多少个micro-step算一个大的step） = 一次step需要的tokens数量 / （一次world forward和backword的token数）
grad_accum_steps = TOTAL_BATCH_SIZE // world_tokens_per_fwdbwd

rank0_print(
    f"每次forward/backward的总token数: {DEVICE_BATCH_SIZE} x {MAX_SEQ_LEN} = {tokens_per_fwdbwd:,}"
)
rank0_print(f"一次world forward和backword的token数: {world_tokens_per_fwdbwd:,}")
rank0_print(
    f"每个 step（做一次 optimizer.step 前）全局累计看到的 token 总数 {TOTAL_BATCH_SIZE:,} => gradient accumulation steps: {grad_accum_steps}"
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
rank0_print(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
rank0_print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# ==========================
# 计算训练步数（用 data:param ratio）
# ==========================

# 这次训练理想情况下总共要喂多少 token 给模型（target_tokens）
target_tokens = TARGET_PARAM_DATA_RATIO * num_params
# （理想情况下）总token数 除以 每个step看到的token数 = 总共要训练多少步
num_iterations = target_tokens // TOTAL_BATCH_SIZE
assert num_iterations > 0
# 真正会训练的总 token 数 = 每步 token × step 数
total_tokens = TOTAL_BATCH_SIZE * num_iterations

rank0_print(f"Calculated number of iterations: {num_iterations:,}")
rank0_print(f"Total training tokens: {total_tokens:,}")
rank0_print(
    f"Tokens : Params ratio: {TOTAL_BATCH_SIZE * num_iterations / num_params:.2f}"
)
rank0_print(
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
tokens_dir = os.path.join(base_dir, "data", "tokenized_data")

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

# 学习率是根据 step（全局训练步数）动态变化的，分三个阶段：
# warmup 段：从 step=0 走到 warmup_iters-1
# 中间 plateau 段：学习率保持不变
# warmdown 段：从 num_iterations - warmdown_iters 到最后一步，慢慢把学习率降下来
def get_lr_multiplier(it: int) -> float:
    warmup_iters = round(WARMUP_RATIO * num_iterations)
    warmdown_iters = round(WARMDOWN_RATIO * num_iterations)
    # lr 从接近 0 线性涨到 1
    if warmup_iters > 0 and it < warmup_iters:
        return (it + 1) / warmup_iters
    # 保持不变
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    # lr 从 1 线性降到 FINAL_LR_FRAC
    else:
        progress = (num_iterations - it) / max(warmdown_iters, 1)
        return progress * 1.0 + (1 - progress) * FINAL_LR_FRAC


# Muon momentum 也是根据 step 变化的，从 0.85 线性升到 0.95
def get_muon_momentum(it: int) -> float:
    frac = min(it / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


# ==========================
# 训练循环
# ==========================


# 准备 checkpoint 目录
output_dirname = MODEL_TAG if MODEL_TAG else f"d{DEPTH}"
checkpoint_dir = os.path.join(base_dir, "weights", "base_checkpoints", output_dirname)
if rank == 0:
    os.makedirs(checkpoint_dir, exist_ok=True)

# 所有其他rank等待 rank0创建好文件夹 然后在同时开始训练
torch.distributed.barrier()


step = 0
min_val_bpb = float("inf")
smooth_train_loss = 0.0
total_training_time = 0.0
max_steps = num_iterations

while step < max_steps:
    # 所以已经消耗的 FLOPs：
    flops_so_far = num_flops_per_token * TOTAL_BATCH_SIZE * step

    # ------- eval on val set -------
    # # step=0 先评估一次，然后每 EVAL_EVERY 步评估
    if step % EVAL_EVERY == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = EVAL_TOKENS // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN * world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        rank0_print(f"[Eval] Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        min_val_bpb = min(min_val_bpb, val_bpb)
        model.train()

    # ------- 单步训练 -------
    synchronize()
    t0 = time.time()

    # 梯度累积 micro steps， 结束后，参数的p.grad上会累积好完整的梯度
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        # 存一个loss的副本，用于日志打印
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        # micro step: 多次 loss.backward() 调用，每次加在同一个 p.grad 上，直到 optimizer.step(), 大step 结束后才更新参数
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)


    # grad_norm 是所有梯度的L2范数，是一个标量，.clip_grad_norm_ 会裁剪梯度避免爆炸
    if GRAD_CLIP > 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            orig_model.parameters(), GRAD_CLIP
        )
        grad_norm = grad_norm_tensor.item()
    else:
        grad_norm = None

    # 根据step来调整lr
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    
    # 根据step来调整 muon momentum
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum

    # 更新参数
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    synchronize()
    t1 = time.time()
    dt = t1 - t0

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    flops_per_sec = num_flops_per_token * TOTAL_BATCH_SIZE / dt
    promised_flops_per_sec_h100 = 989e12 * world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100

    if step > 10:
        total_training_time += dt

    grad_str = f" | grad_norm: {grad_norm:.4f}" if grad_norm is not None else ""
    rank0_print(
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
# 训练结束后的 final eval + checkpoint
# ==========================

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        rank0_print(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        rank0_print(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        rank0_print(f"Saved optimizer state to: {optimizer_path}")


model.eval()
val_loader = build_val_loader()
eval_steps = EVAL_TOKENS // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN * world_size)
with autocast_ctx:
    val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
rank0_print(f"[Final Eval] Step {step:05d} | Validation bpb: {val_bpb:.4f}")
min_val_bpb = min(min_val_bpb, val_bpb)


meta_data = {
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

# 保存模型权重和优化器状态
save_checkpoint(
    checkpoint_dir,
    step,
    orig_model.state_dict(),  # 保存模型参数
    [opt.state_dict() for opt in optimizers],  # 保存优化器状态
    meta_data,
    rank=rank,
)

# ==========================
# 训练结束总结
# ==========================

rank0_print(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
rank0_print(f"Total training time: {total_training_time/60:.2f}m")
rank0_print(f"Minimum validation bpb: {min_val_bpb:.4f}")

dist.destroy_process_group()
