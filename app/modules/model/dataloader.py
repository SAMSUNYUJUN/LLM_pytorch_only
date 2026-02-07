from collections import deque

import torch
import pyarrow.parquet as pq

from app.modules.utils.utils import get_dist_info
from app.modules.fetch_data.dataset import list_parquet_files
from app.modules.tokenizer.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        # 一个parquet文件中有多个row group，每个row group包含多行文本数据。
        parquet_paths = list_parquet_files()
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

        # checkpoint 里记录的“上次训练中断的大致位置”
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None

        # 本次“重启训练”里的第一圈，用 resume；后面的圈就当正常从 0 开始
        first_pass = True

        while True:  # 无限 epoch / 无限 pass
            # 本次 pass 的起始文件：
            #   - 第一次 pass：从 resume_pq_idx 开始（接着上次中断处往后走）
            #   - 后续 pass：从 0 开始（完整扫一圈数据）
            pq_idx = resume_pq_idx if first_pass else 0

            while pq_idx < len(parquet_paths):  # 遍历本轮要看的所有 parquet 文件
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)

                # 只有在“第一次 pass 且在恢复的那个文件上”才用 resume_rg_idx
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    # resume_rg_idx 是“上次看到的 row group index”，我们跳到它后面的那块
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1  # +1 保证不重复上次那块
                    rg_idx = base_idx * ddp_world_size + ddp_rank

                    # 如果这一跳直接跳出这个文件的 row group 范围，
                    # 说明本 rank 在这个文件里已经没数据可读了，直接下一个文件
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue

                    # 用过一次 resume_rg_idx 之后就作废，后面不再用
                    resume_rg_idx = None
                else:
                    # 正常 DDP 切分：每个 rank 从自己的 ddp_rank 起步，间隔 world_size
                    rg_idx = ddp_rank

                # 遍历当前 parquet 文件中属于本 rank 的所有 row group
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column("text").to_pylist()  # 每个 row group 是很多行 text

                    # 再按 tokenizer_batch_size 切成更小的 doc batch
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i : i + tokenizer_batch_size], (pq_idx, rg_idx)

                    # 下一个属于本 rank 的 row group（交错分配）
                    rg_idx += ddp_world_size

                # 当前 parquet 文件读完，切到下一个文件
                pq_idx += 1

            # 第一圈跑完之后，以后就不再用 resume_pq_idx/resume_rg_idx 了
            first_pass = False
    batches = document_batches()

    # Now emit batches of tokens.
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # in PyTorch, long=int64
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
