"""
分布式AdamW，类似ZeRO-2，即分片优化器状态和梯度归约。

AdamW优化器解释：
pytorch中AdamW的超参数：
torch.optim.AdamW(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)
-lr（学习率 α）：决定整体步子大小，越大走得越快，也越容易发散。Adam 系列通常 1e-3, 3e-4, 1e-4 这一档。
-betas = (β1, β2)：β1：一阶动量的衰减系数（对梯度本身做滑动平均）一般 0.9 左右 → 表示“新的梯度占 10%，旧的记忆占 90%”； β2：二阶动量的衰减系数（对梯度平方做滑动平均）一般 0.999 / 0.98 → 平滑每一维梯度的方差。
-eps（ε）防止分母为 0 的小常数，一般 1e-8。数值非常小，只在梯度极小或刚开始时起一点稳定作用。
- weight_decay（权重衰减 λ）就是 L2 正则强度。典型范围：0 ~ 0.1，大模型里常见 0.01 左右。

一阶动量： mt​=β1 * ​mt−1 ​+(1−β1​)gt​
mt初始值为0。mt是滑动平均。直观类比：你在推一个球下坡：当前梯度 g_t = 当前“坡的方向”，m_t = 球的速度（有惯性，不会立刻掉头），β1 越大，惯性越强，越“滑”。来一个 1D 具体小例子：

β1 = 0.9； 𝑚0=0； 三个 step 的梯度：g₁ = 1； g₂ = 1 ；g₃ = -1（方向突然反了）
我们算一下：
t = 1：m1=0.9⋅0+0.1⋅1=0.1
t = 2：m2 =0.9⋅0.1+0.1⋅1=0.09+0.1=0.19
t = 3（梯度变成 -1）：m3=0.9⋅0.19+0.1⋅(-1)=0.171-0.1=0.071
单看梯度：[1, 1, -1] → 第三步突然反向, 动量 m：[0.1, 0.19, 0.071] → 只是慢慢往下减，还没有立刻变成负数, 这就说明：动量不是“当前这一步的梯度”，是“这段时间梯度的平均趋势”，帮你平滑震荡 + 提高收敛效率。

在 AdamW 中，真正更新参数的不是 g_t，而是 bias-correct 后的 m̂_t = m_t / (1 - β1^t)，因为 m_t 刚开始时偏向于 0，需要校正。随着 t 变大，(1 - β1^t) 趋近于 1，校正作用减小。
	​
二阶动量： vt​=β2 * ​vt−1 ​+(1−β2​)gt​^2
vt 初始值为 0。vt 也是滑动平均，但它是对梯度平方的滑动平均，表示“每一维梯度的方差”。类比上面的例子：
β2 = 0.999； 𝑣0=0； 三个 step 的梯度：g₁ = 1； g₂ = 1 ；g₃ = -1
t = 1：v1=0.999⋅0+0.001⋅1²=0.001
t = 2：v2=0.999⋅0.001+0.001⋅1²=0.000999+0.001=0.001999
t = 3：v3=0.999⋅0.001999+0.001⋅(-1)²=0.001997001
可以看到，v_t 变化非常慢，表示“这几步梯度的方差大概在 0.002 左右”。为什么要关注方差？因为：如果某一维梯度方差很大，说明这一维梯度不稳定，可能忽大忽小，直接用它来更新参数会不稳，所以我们要“抑制”它的更新。
在 AdamW 中，真正用来更新参数的不是 v_t，而是 bias-correct 后的 v̂_t = v_t / (1 - β2^t)，同样是为了校正刚开始时的偏差。

参数更新公式：
p - lr * m_hat_t / (sqrt(v_hat_t) + eps)
我们可以发现，这个近似于在给lr做调整，调整的是参数的平均值和方差：我们用动量的方式来估计梯度的均值 m̂_t 和方差 v̂_t，这样比直接算所有历史梯度的均值和方差要高效得多。而且可以让当前当前的平均和方差更关注近期的梯度变化，更灵活地适应当前的优化形势。
"""
import torch
import torch.distributed as dist
from torch import Tensor


class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction
    """

    # param_groups是矩阵参数和学习率[{"params": [W_q, W_k, ...], "lr": 0.02, ...},  {"params": [embedding], "lr": 0.2, ...},...]
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        # rank指进程，一般来说一个rank对应一张gpu
        # rank会动态检测返回当前进程的rank id， 如果在第一张卡上，rank就是0，第二张卡上rank就是1。
        rank = dist.get_rank()
        # 进程总数
        world_size = dist.get_world_size()
        #准备三个列表：
        #reduce_scatter_futures：存每个 reduce_scatter 的 future，后面要 .wait()；
        #all_reduce_futures：存每个 all_gather 的 future；
        #grad_slices：存每个参数对应的 梯度切片（每个 rank 拿自己那一片）。
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []

        # 第一轮循环，对每个参数的梯度做 reduce_scatter 操作，把结果放到 grad_slices 里
        # reduce_scatter 是把一个张量按第0维切分成 world_size 份，然后每个 rank 拿其中一份，并对所有 rank 的对应份做 reduce 操作（这里是求和再平均）
        # 便利param_groups里的每个参数组
        # 这里有一个异步计算，可以先循环把所有的 reduce_scatter_tensor 操作都发起，然后再等它们完成
        for group in self.param_groups:
            # 得到该组的参数列表
            params: list[Tensor] = group["params"]
            # 便利该组的每个参数
            for base_i in range(len(params)):
                # 得到该参数的梯度
                grad = params[base_i].grad
                # 按 rank 数量切分梯度
                rank_size = grad.shape[0] // world_size
                # 准备一个空张量的形状是梯度分成两半后的形状 （两张gpu）
                grad_slice = torch.empty_like(grad[:rank_size])
                # dist.reduce_scatter_tensor 是个全局通信操作，相当于一个分配员。
                # 它会把所有 rank 的 grad 先按元素 AVG；再把 avg_grad 分成两段：前一半行分配给 rank 0，后一半行分配给 rank 1；让他们分别进行计算。
                # 然后分别存在各自rank的 grad_slice 里，也就是说，每张卡上有自己单独的grad_slices.
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                # 把该参数对应的梯度切片存起来
                grad_slices.append(grad_slice)

        # 第二轮双循环：等待梯度通信完成 → 对每片参数做 AdamW 更新 → all_gather 拼回参数
        idx = 0
        # 依旧遍历每个参数组
        for group in self.param_groups:
            # 取出 AdamW 的超参
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            # 取出该组的参数
            params = group['params']

            # 遍历该组的每个参数
            for base in range(len(params)):
                # 等待对应的 reduce_scatter 操作完成，拿到该 rank 的梯度切片
                reduce_scatter_futures[idx].wait()
                # 切出当前 rank 自己负责的参数块 p_slice，并只更新这部分参数
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

                # 取学习率、状态、梯度切片
                # lr：这一组参数的学习率，如果 p 上挂了 p.lr_mul，再乘一个缩放（比如有的层用更小 lr）
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                # state = self.state[p]：这是每个参数一个字典，用来存：step（第几步）；exp_avg（一阶动量）；exp_avg_sq（二阶动量）
                state = self.state[p]
                # 取出对应的梯度切片
                g_slice = grad_slices[idx]

                # State init
                # 第一次 step 时，state 是空的，走初始化：
                # step = 0
                # exp_avg、exp_avg_sq 的 shape 和 p_slice 一样：
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)

                # 取出当前step的一阶、二阶动量
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                # 更新step
                state['step'] += 1
                t = state['step']

                # weight decay
                # 在p_slice上做weights衰减
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)

                # update running averages
                # 更新一阶、二阶动量
                # 一阶动量 mt​=β1 * ​mt−1 ​+(1−β1​)gt​
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                # 二阶动量 vt​=β2 * ​vt−1 ​+(1−β2​)gt​^2
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)

                # ---- bias correction 分母 ----
                # m_hat_t​=mt​/(1−β1 ** t​)， 这里只计算了分母部分
                bias1 = 1 - beta1 ** t
                # v_hat_t​=vt​/(1−β2 ** t​)， 这里只计算了分母部分
                bias2 = 1 - beta2 ** t

                #   p ← p - lr * m_hat_t / (sqrt(v_hat_t) + eps)
                #   = p - lr * [m_t / (1 - beta1^t)] / (sqrt(v_t / (1 - beta2^t)) + eps)
                m_hat = exp_avg / bias1           # m̂_t
                v_hat = exp_avg_sq / bias2        # v̂_t
                denom = v_hat.sqrt().add_(eps)    # sqrt(v̂_t) + eps
                step_size = lr                   # lr
                update = m_hat / denom * step_size  # lr * m̂_t / (sqrt(v̂_t) + eps)

                # 更新参数 p ← p - lr * m_hat_t / (sqrt(v_hat_t) + eps)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1

                # all gather updated param slice back to full param
                all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        # 等待所有 all_gather 操作完成
        torch.futures.collect_all(all_reduce_futures).wait()
