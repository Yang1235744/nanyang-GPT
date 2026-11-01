# nanyang-GPT —— 从零自研的字节级对话大模型

decoder-only Transformer，**不依赖任何预训练权重**：tokenizer、模型、训练全部从零实现。
原生支持工具调用标记，可与 [nanyang-Helios1.5](https://github.com/Yang1235744/nanyang-Helios1.5) 捆绑 —— 先聊天，需要时触发生成图片 / 视频。

## 特性

| 项 | 说明 |
| --- | --- |
| 架构 | decoder-only Transformer（pre-norm 因果注意力 + 权重共享输出头） |
| 参数量 | 305 M（dim 1024 · depth 24 · heads 16） |
| Tokenizer | 自研 UTF-8 字节级（词表 **259**：0-255 字节 + PAD/BOS/EOS），中英文原生通吃 |
| 上下文 | 2048 tokens |
| 工具调用 | `<\|gen_image\|>` / `<\|gen_video\|>` 标记，`parse_tool()` 解析 |
| 权重体积 | fp32 约 1.2 GB |

## 快速开始

```bash
git clone https://github.com/Yang1235744/nanyang-GPT.git
cd nanyang-GPT && pip install -r requirements.txt

# 语料：把对话文本放进 data/corpus/*.txt
# 训练
python -m nanyang_gpt.train --config configs/nanyang_gpt_medium.yaml

# 聊天
python -m nanyang_gpt.chat --ckpt runs/nanyang_gpt/latest_gpt.pt
```

## 两种格式 · 两个通道

同一个模型，两种格式，各走各的通道：

| | 通道一：Ollama | 通道二：通用 |
| --- | --- | --- |
| 格式 | **GGUF**（gpt2 架构家族） | **PyTorch checkpoint**（.pt） |
| 适合 | Ollama / llama.cpp 等本地运行时 | 任意 PyTorch 项目、Helios 1.5 捆绑、二次开发 |
| 获取 | 训练后自行转换（见下），或直接下载现成 GGUF | `git clone` 本仓库 + 下载现成 .pt |

现成权重（`.pt` 与 `.gguf` 双格式，永久直链，无需注册）：

```
https://share.fnnas.net/s/90de87a72789493597
```

### 通道一：转成 GGUF 接 Ollama

```bash
pip install gguf
python -m nanyang_gpt.convert_gguf --ckpt runs/nanyang_gpt/latest_gpt.pt \
    --out nanyang-gpt-f16.gguf --ftype f16

ollama create nanyang-gpt -f Modelfile
ollama run nanyang-gpt
```

南阳 GPT 的拓扑与 GPT-2 一致（可学习位置嵌入、pre-norm、qkv 合一投影、GELU MLP、
嵌入共享），转换器会按 gpt2 架构落盘，tokenizer 用字节级词表（0-255 = UTF-8 字节），
llama.cpp 的 BPE 编解码天然兼容。

### 通道二：PyTorch 通用

`.pt` checkpoint 就是标准 `state_dict`，任何 PyTorch 项目 `torch.load` 即用；
与 [nanyang-Helios1.5](https://github.com/Yang1235744/nanyang-Helios1.5) 内置的
`helios/models/nanyang_gpt.py` 同架构同配置，权重直接互换。

## 工具调用

语料里教会模型在该生成时输出标记，运行时解析：

```
用户：帮我画一只雪地里的红狐
助手：好的，这就为你生成。<|gen_image|>雪地里的红狐
```

```python
from nanyang_gpt import parse_tool
text, tool, prompt = parse_tool(reply)   # -> ("好的，这就为你生成。", "image", "雪地里的红狐")
```

## 与 Helios 1.5 捆绑

本仓库与 Helios 1.5 内置的 `helios/models/nanyang_gpt.py` **同架构、同配置**，checkpoint 可以直接互换：

```bash
python -m helios.sample --gpt-ckpt runs/nanyang_gpt/latest_gpt.pt ...
```

## 路线图

- [ ] GGUF 转换版 —— 接入 Ollama 等本地运行时，一行命令拉取即聊
- [ ] 更大规模语料与指令微调
- [ ] 更长的上下文窗口

## 许可

MIT
