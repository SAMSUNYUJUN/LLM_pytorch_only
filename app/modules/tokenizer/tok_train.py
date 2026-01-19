"""
Train a tokenizer using HuggingFace Tokenizers (BPE).
GPT-4-style pre-tokenization (same as tokenizer.py).
"""

import os
import sys
import time
import argparse
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.modules.tokenizer.tokenizer import HuggingFaceTokenizer
from app.modules.utils.utils import get_base_dir
from app.modules.fetch_data.dataset import parquets_iter_batched


# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
parser.add_argument(
    "--max-chars",
    type=int,
    default=10_000_000_000,
    help="Maximum characters to train on (default: 10B)",
)
parser.add_argument(
    "--doc-cap",
    type=int,
    default=10_000,
    help="Maximum characters per document (default: 10,000)",
)
parser.add_argument(
    "--vocab-size",
    type=int,
    default=32768,
    help="Vocabulary size (default: 32768 = 2^15)",
)
args = parser.parse_args()

print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")


# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc if isinstance(doc, str) else str(doc)
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[: args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return


# -----------------------------------------------------------------------------
# Train the tokenizer

t0 = time.time()
tokenizer = HuggingFaceTokenizer.train_from_iterator(text_iterator(), args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")


# -----------------------------------------------------------------------------
# Save the tokenizer to disk (must be app/weights/tokenizer)

base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "weights", "tokenizer")
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save(tokenizer_dir)


# -----------------------------------------------------------------------------
# Quick inline sanity check

test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text, "Tokenizer sanity check failed (decoded != original)"


# -----------------------------------------------------------------------------
# Cache token_bytes.pt to app/weights/tokenizer/token_bytes.pt

vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())

token_bytes = []
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])  # string representation of a single token
    if token_str in special_set:
        token_bytes.append(0)  # special tokens not counted
    else:
        token_bytes.append(len(token_str.encode("utf-8")))

token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)

print(f"Saved token_bytes to {token_bytes_path}")
