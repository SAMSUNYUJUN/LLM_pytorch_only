import os
import copy

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        bos = self.encode_special("<|bos|>")
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

    def render_conversation(self, conversation, max_tokens: int = 2048):
        """
        将一整个多轮对话转成 token 序列，并生成 supervision mask。

        参数
        ----
        conversation:
            - 可以是 dict: {"messages": [...]}，其中 messages 是一个 list
            - 也可以直接是 list: [{"role": "user"/"assistant"/"system", "content": ...}, ...]
        max_tokens:
            - 最多保留多少个 token（超出会截断，防 OOM）

        返回
        ----
        ids:  List[int]  整个对话的 token id 序列
        mask: List[int]  同长度，0/1：
              - 1 表示这些 token 需要算 loss（通常是 assistant 的回复部分）
              - 0 表示不算 loss（user 提示、系统提示、工具输出等）
        """

        # 归一化成 messages: List[dict]
        if isinstance(conversation, dict):
            messages = conversation.get("messages", [])
        else:
            messages = conversation

        if not messages:
            raise ValueError(f"Conversation has no messages: {conversation}")

        # 处理开头可能存在的 system message：
        # 如果第一条是 system，就把内容拼到后面第一条 user 上
        if messages[0].get("role") == "system":
            msgs = copy.deepcopy(messages)
            if len(msgs) < 2 or msgs[1].get("role") != "user":
                raise AssertionError("System message must be followed by a user message")
            sys_content = msgs[0].get("content", "")
            user_content = msgs[1].get("content", "")
            msgs[1]["content"] = f"{sys_content}\n\n{user_content}"
            messages = msgs[1:]
        # 否则直接用原来的 messages
        # messages = messages

        if len(messages) < 1:
            raise ValueError(f"Conversation has less than 1 message after system merge: {messages}")

        ids: list[int] = []
        mask: list[int] = []

        def add_tokens(token_ids, mask_val: int):
            # 小工具：一边加 token，一边加对应 mask
            nonlocal ids, mask
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # 取特殊 token id
        bos = self.get_bos_token_id()
        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")
        python_start = self.encode_special("<|python_start|>")
        python_end = self.encode_special("<|python_end|>")
        output_start = self.encode_special("<|output_start|>")
        output_end = self.encode_special("<|output_end|>")

        # 开头先加一个 BOS，不算 loss
        add_tokens(bos, 0)

        for i, message in enumerate(messages):
            role = message.get("role", "user")
            content = message.get("content", "")

            # 假定对话严格交替：user, assistant, user, assistant, ...
            must_be_from = "user" if i % 2 == 0 else "assistant"
            if role != must_be_from:
                raise AssertionError(
                    f"Message {i} is from {role!r} but should be from {must_be_from!r}"
                )

            # user 部分：只作为条件，不参与 loss，mask 全是 0
            if role == "user":
                if not isinstance(content, str):
                    raise TypeError("User messages are expected to be plain strings")
                value_ids = self.encode(content)

                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)

            # assistant 部分：模型要学的输出，正文部分 mask = 1
            elif role == "assistant":
                add_tokens(assistant_start, 0)

                # 情况 1：简单字符串
                if isinstance(content, str):
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)

                # 情况 2：多段结构，比如 [{"type": "text", ...}, {"type": "python", ...}, ...]
                elif isinstance(content, list):
                    for part in content:
                        part_type = part.get("type", "text")
                        part_text = part.get("text", "")

                        value_ids = self.encode(part_text)

                        if part_type == "text":
                            # 普通文本 -> 需要监督，mask=1
                            add_tokens(value_ids, 1)

                        elif part_type == "python":
                            # 代码调用 -> 用 <|python_start|> 和 <|python_end|> 包裹，mask=1
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)

                        elif part_type == "python_output":
                            # 代码输出 -> 推理时由工具给出，不需要模型预测，mask=0
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)

                        else:
                            raise ValueError(f"Unknown part type: {part_type!r}")

                else:
                    raise TypeError(f"Unknown assistant content type: {type(content)}")

                add_tokens(assistant_end, 1)

            else:
                raise ValueError(f"Unknown role in messages: {role!r}")

        # 最后截断，避免过长导致显存爆炸
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation):
        """
        用于 RL / 采样场景：把一整个对话渲染成 prompt，
        最后以 Assistant 开头，等待模型继续生成。

        不在这里做 max_tokens 截断，长度控制交给调用方。
        """

        # 归一化成 {"messages": [...]} 结构并深拷贝
        if isinstance(conversation, dict):
            conv = copy.deepcopy(conversation)
            messages = conv.get("messages", [])
        else:
            messages = copy.deepcopy(conversation)
            conv = {"messages": messages}

        if not messages:
            raise ValueError(f"Conversation has no messages: {conversation}")

        # 最后一条必须是 assistant
        if messages[-1].get("role") != "assistant":
            raise AssertionError("Last message must be from the Assistant")

        # 删掉最后一条 assistant，只用前面的对话作为条件
        messages.pop()

        # 用前面的对话跑一遍 render_conversation（使用默认 max_tokens）
        ids, _ = self.render_conversation(conv)

        # 在末尾追加 assistant_start，提示模型“从这里开始接着写 Assistant 回复”
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)

        return ids

def get_tokenizer():
    from app.modules.utils.utils import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "weights", "tokenizer")
    return HuggingFaceTokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    import torch
    from app.modules.utils.utils import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "weights", "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
