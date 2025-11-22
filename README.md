# Python 环境设置（uv + PyTorch + tokenizers）

本项目使用 [uv](https://astral.sh) 管理虚拟环境和依赖，主要依赖为：

- `torch`（PyTorch）
- `tokenizers`

## 1. 安装 uv

如果你还没安装 uv，可以在终端中运行（macOS / Linux）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```