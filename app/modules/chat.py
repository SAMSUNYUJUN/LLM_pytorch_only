"""
python -m app.modules.chat \
  --ckpt-relpath base_checkpoints/d32/model_076800.pt \
  --device

"""
import os
import sys
import argparse
import torch
from contextlib import nullcontext

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.modules.utils.utils import compute_init, autodetect_device_type, get_base_dir
from app.modules.model.engine import Engine
from app.modules.utils.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('--ckpt-relpath', type=str, required=True, help='Path under app/weights to a checkpoint dir or model_XXXXXX.pt (e.g. base_checkpoints/d32 or base_checkpoints/d32/model_076800.pt).')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
args = parser.parse_args()

# Init the model and tokenizer

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
if args.ckpt_relpath:
    from app.modules.utils.checkpoint_manager import build_model, find_last_step
    base_dir = get_base_dir()
    weights_root = os.path.join(base_dir, "weights")
    full_path = os.path.join(weights_root, args.ckpt_relpath)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Checkpoint path not found: {full_path}")
    if os.path.isfile(full_path):
        fname = os.path.basename(full_path)
        import re
        m = re.search(r"model_(\d+)\.pt", fname)
        if not m:
            raise ValueError(f"Cannot parse step from filename: {fname}")
        step = int(m.group(1))
        checkpoint_dir = os.path.dirname(full_path)
    else:
        checkpoint_dir = full_path
        step = find_last_step(checkpoint_dir)
    model, tokenizer, meta = build_model(checkpoint_dir, step, device, phase="eval", allow_missing_c_gate=True)
elif args.source == "base_ckpt":
    from app.modules.utils.checkpoint_manager import load_model_from_dir
    base_dir = get_base_dir()
    ckpt_root = os.path.join(base_dir, "weights", "base_checkpoints")
    model, tokenizer, meta = load_model_from_dir(ckpt_root, device, phase="eval", model_tag=args.model_tag, step=args.step)
else:
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# Special tokens for the chat state machine
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# Create Engine for efficient generation
engine = Engine(model, tokenizer)

print("\nSamChat")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
print("-" * 50)

conversation_tokens = [bos]

while True:

    if args.prompt:
        # Get the prompt from the launch command
        user_input = args.prompt
    else:
        # Get the prompt interactively from the console
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    # Handle special commands
    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue

    if not user_input:
        continue

    # Add User message to the conversation
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)

    # Kick off the assistant
    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 256,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    with autocast_ctx:
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0] # pop the batch dimension (num_samples=1)
            response_tokens.append(token)
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)
    print()
    # we have to ensure that the assistant end token is the last token
    # so even if generation ends due to max tokens, we have to append it to the end
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    # In the prompt mode, we only want a single response and exit
    if args.prompt:
        break
