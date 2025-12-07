'''
Transformer 里最“重”的参数，就是一堆线性层 / attention Wq, Wk, Wv, Wo, MLP 的权重矩阵。

这些矩阵的“坏情况”不是单个元素太大，而是某几个方向（奇异向量）被放大太多（谱范数很大），导致网络 Lipschitz 常数飙升，训练不稳；

AdamW 的二阶动量是 per-element 的，看不见“奇异向量”这种全局结构：

你可以把同一奇异向量上的能量拆成很多参数，AdamW 只知道“每一格的均方”。

Muon 的做法是：

先用一阶动量汇总“整体梯度矩阵的方向”；

然后保证更新矩阵在谱范数意义下是“等长”的，奇异值控制在 ≈ [0.5, 1.5]：

不会朝某个奇异方向更新过猛，

也不会让某些方向完全没更新。

结果就是：

对每个线性层来说，Muon 在 适应“整体几何”和“最坏方向” 上比 AdamW 的对角二阶动量更聪明。
'''
import torch
from torch import Tensor
import torch.distributed as dist
from typing import Iterable, Tuple, Optional
from torch import nn

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    通过 Newton-Schulz 方法计算矩阵 G 的 -1/2 次幂的近似。
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # 矩阵归一化，最大的值是1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # 把 X 迭代多次，让其变的像正交矩阵，也就是把过大的奇异值压缩到，过小的奇异值拉升。
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    params: 可迭代的参数列表
    lr: 学习率
    momentum: 动量系数
    nesterov: 是否使用 Nesterov 动量
    ns_steps: Newton-Schulz 迭代步数
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                # 获取梯度
                g = p.grad
                assert g is not None
                # 获取状态
                state = self.state[p]

                # 初始化动量缓冲区，全是0的矩阵，shape和梯度矩阵一样
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                
                # 更新动量缓冲区
                buf: Tensor = state["momentum_buffer"]

                # buf = m⋅buf_old+(1-m)⋅g = 0.95*buf_old + 0.05*g
                # 这里等价于adamW的一阶动量
                # 每次都按固定权重把旧值和新值混合，旧值的系数会按幂次衰减,展开多步（t 次）：buf_t = (1 - m) * g_t + m * (1 - m) * g_{t-1} + m^2 * (1 - m) * g_{t-2} + … + m^t * buf_0
                buf = buf.mul_(group["momentum"]).add_(g, alpha=1 - group["momentum"])
                # 如果使用 Nesterov
                if group["nesterov"]:
                    # g = m⋅buf + (1-m)⋅g = 0.95*buf + 0.05*g
                    g = g.mul_(1 - group["momentum"]).add_(buf, alpha=group["momentum"])
                else:
                    g = buf

                # 使用 Newton-Schulz 让gradient矩阵正交化
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # p = p - lr * g * sqrt(max(1, rows/cols))
                # sqrt(max(1, rows/cols)) 是对非方阵的步长修正：当矩阵行数大于列数时放大步长，行列比不大于 1 时不放大。常见于做 factored 二阶/动量统计时，用这个比例补偿形状带来的缩放差异，使更新幅度更平衡
                p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)


class DistMuon(torch.optim.Optimizer):
    """
    分布式 Muon 优化器。
    params: 可迭代的参数列表
    lr: 学习率
    momentum: 动量系数
    nesterov: 是否使用 Nesterov 动量
    ns_steps: Newton-Schulz 迭代步数
    """

    # muon期待的输入params是一个二维矩阵列表，例如[W_q, W_k, W_v, W_out, ...]
    def __init__(self, 
                 params: Iterable[torch.nn.Parameter],
                 lr: float = 0.02,
                 momentum: float = 0.95,
                 nesterov: bool = True,
                 ns_steps: int = 5):
        
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)

        # 按形状把参数分组
        param_groups = []
        shape2idx = {}  # shape -> index in param_groups
        for p in params:
            # 所有参数必须是二维矩阵
            if p.ndim != 2:
                raise ValueError(f"Expected 2D tensor, got {p.ndim}D with shape {tuple(p.shape)}")
            
            shape = tuple(p.shape)  # torch.Size -> tuple，方便当 key
            if shape not in shape2idx:
                # 第一次遇到这个形状，创建一个新的 group
                shape2idx[shape] = len(param_groups)
                param_groups.append(dict(
                    params=[p],
                    shape=shape, 
                ))
            else:
                # 已经有这个形状的 group，往里 append
                idx = shape2idx[shape]
                param_groups[idx]["params"].append(p)

        # param_groups.sort(
        #     key=lambda g: g["shape"][0] * g["shape"][1],
        #     reverse=True,
        # )

        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # params必须有grad
        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), "All params must have grads"
        
        # 异步发起“平均每个rank上的grad”操作的请求列表
        all_reduce_futures = []

        # 便利相同shape的参数组
        for group in self.param_groups:
            params = group["params"]
            # 这个 group 里如果从头到尾都不需要 padding，就不会创建 zero_buffer
            zero_buffer = None
            # 对这一组参数的梯度做平均，并按照参数编号轮流分配给不同的 rank，
            # 每个参数的平均梯度只存在一个 owner rank 上。
            for i in range(0, len(params), world_size):
                owner_idx = i + rank # 这个 block 中，本 rank 拥有的参数索引

                if i + world_size <= len(params):
                # 完整 block，不需要 padding
                    rs_input = [p.grad for p in params[i:i + world_size]]
                    rs_output = params[owner_idx].grad
                else:
                # 不足 world_size，需要 padding，才会用到 zero_buffer
                    if zero_buffer is None:
                        # 懒创建：只在第一次需要时建
                        zero_buffer = params[0].grad.new_zeros(params[0].grad.shape)
                    rs_input = [p.grad for p in params[i:]]
                    pad = world_size - len(rs_input)
                    rs_input.extend([zero_buffer] * pad)

                    # owner rank 可能超出范围，超出范围的都用0来接收
                    if owner_idx < len(params):
                        rs_output = params[owner_idx].grad
                    else:
                        rs_output = torch.empty_like(zero_buffer)     
                # 把 reduce_scatter 的平均操作异步发起，发给rs_output对应的rank。
                # 这里reduce_scatter方法把一个rs_input list, 给每个位置做平均，但是每个rank只能拿到自己对应index的那一份平均grad
                work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
                # 现在第 owner rank index 上的 rs_output 就是该参数的平均梯度
                # all_reduce_futures的列表大小是 ceil(len(params) / world_size)
                all_reduce_futures.append(work)

        future_idx = 0
        # 异步发起“广播更新后参数”操作的请求列表
        all_gather_futures = []
        # 第二轮双循环：等待梯度通信完成 → 对每片参数做 Muon 更新 → all_gather 拼回参数
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = None
            for i in range(0, len(params), world_size):
                owner_idx = i + rank 
                # 异步得到 reduce_scatter 的结果
                all_reduce_futures[future_idx].wait() 
                future_idx += 1
                # 判断当前 owner_idx 是否在参数范围内，这里防止便利到pad的部分
                if owner_idx < len(params):
                    p = params[owner_idx]
                    # 得到平均梯度
                    g = p.grad  
                    state = self.state[p]
                    # 初始化动量缓冲区，全是0的矩阵，shape和梯度矩阵一样
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    # 更新动量缓冲区，一阶动量
                    buf = buf.mul_(group["momentum"]).add_(g, alpha=1 - group["momentum"])
                    if group["nesterov"]:
                        g = g.mul_(1 - group["momentum"]).add_(buf, alpha=group["momentum"])
                    else:
                        g = buf
                    # 使用 Newton-Schulz 让gradient矩阵正交化
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    # 参数更新
                    p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

                    all_gather_input_tensor = params[owner_idx]
                    all_gather_output_list = params[i:i + world_size]

                else:
                    # dist.all_gather(output_tensor_list, input_tensor) 这个 API 要求：len(output_tensor_list) == world_size
                    # 所以需要pad到长度 world_size
                    if zero_buffer is None:
                        zero_buffer = params[0].grad.new_zeros(params[0].grad.shape)
                    all_gather_input_tensor = zero_buffer
                    all_gather_output_list = params[i:]
                    pad = world_size - len(all_gather_output_list)
                    all_gather_output_list.extend([zero_buffer.clone() for _ in range(pad)])
                
                # all_gather 例子：假如world_size = 4
                # all_gather_output_list = [p0, p1, p2, p3] #
                # all_gather_input_tensor= params[owner_idx]  # 每个 rank 不同：rank0->p0, rank1->p1, rank2->p2, rank3->p3
                # all_gather后，所有rank的 p0都会变成 rank0的参数，p1都会变成rank1的参数，依次类推。
                # zero_buffer由于并不在params参数列表里，所以就算被传播了，也不会影响训练。
                work = dist.all_gather(all_gather_output_list, all_gather_input_tensor, async_op=True).get_future()
                all_gather_futures.append(work)

        # Wait for all work to finish
        torch.futures.collect_all(all_gather_futures).wait()

if __name__ == "__main__":
    # 1. 初始化分布式进程组（用 torchrun 启动的话，env:// 会从环境变量里读 RANK / WORLD_SIZE / MASTER_*）
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # 2. 每个 rank 绑定一块 GPU：rank 0 -> cuda:0, rank 1 -> cuda:1
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(f"[rank {rank}] init done, world_size = {world_size}", flush=True)
    # 3. 固定随机种子（每个 rank 不同，方便验证 reduce_scatter 确实在做事）
    torch.manual_seed(123+rank)
    # 4. 建一个简单模型：4096 -> 4096，只要 2D 参数，避免 bias 这种 1D
    model = nn.Sequential(nn.Linear(4096, 4096, bias=False),nn.Linear(4096, 4096, bias=False),nn.Linear(4096, 1024, bias=False),nn.Linear(1024, 4096, bias=False)).to(device)
    # 5. 构造 DistMuon，只传入 2D 参数
    optim = DistMuon(model.parameters(), lr=0.01)

    # 6. 一个非常简单的训练循环：随机输入、随机 target，做 MSE
    num_steps = 20
    batch_size = 8

    for step in range(num_steps):
        # 清梯度（用 optimizer 自带的 zero_grad）
        optim.zero_grad()
        # 随机生成输入和目标
        x = torch.randn(batch_size, 4096, device=device)
        target = torch.randn(batch_size, 4096, device=device)
        # 前向
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        # 反向传播，给每个参数矩阵挂上grad
        loss.backward()
        # 分布式 Muon 更新参数（内部会用 reduce_scatter + all_gather）
        optim.step()
        # 每隔几步打印一下每个 rank 的 loss 和权重范数，看看是否一致
        if step % 5 == 0:
            with torch.no_grad():
                # 只取第一个参数（就是 Linear 的 weight）
                weight = next(model.parameters())
                w_norm = weight.norm().item()
            print(
                f"[rank {rank}] step {step:02d} | loss = {loss.item():.4f} | "
                f"weight_norm = {w_norm:.4f}",
                flush=True,
            )

        # 同步一下，避免输出太乱
        dist.barrier()

    # 训练结束后，再检查一次两个 rank 上的 weight 是否一致
    with torch.no_grad():
        w = next(model.parameters()).clone()
    # 收集所有 rank 的 weight norm，确认大家一样
    w_norm = torch.tensor([w.norm().item()], device=device)
    all_norms = [torch.zeros_like(w_norm) for _ in range(world_size)]
    dist.all_gather(all_norms, w_norm)

    if rank == 0:
        print("\n[rank 0] Final weight norms from all ranks:")
        for r, n in enumerate(all_norms):
            print(f"  rank {r}: {n.item():.6f}", flush=True)

    dist.destroy_process_group()

