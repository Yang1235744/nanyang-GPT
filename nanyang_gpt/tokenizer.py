"""字节级 tokenizer：UTF-8 字节 + 3 个特殊符号，词表 259，中英文原生通吃。

不依赖 sentencepiece / 预训练 embedding，任何文本都能编解码。
"""

PAD_ID, BOS_ID, EOS_ID = 256, 257, 258
VOCAB_SIZE = 259


def encode_text(text: str, max_len: int) -> list[int]:
    """文本 -> 定长 token 序列（BOS + UTF-8 字节 + EOS + PAD 补齐）。"""
    ids = [BOS_ID] + list(text.encode("utf-8"))[: max_len - 1] + [EOS_ID]
    return ids + [PAD_ID] * (max_len - len(ids))


def encode_stream(text: str) -> list[int]:
    """文本 -> 变长 token 流（BOS + 字节 + EOS），用于训练拼接。"""
    return [BOS_ID] + list(text.encode("utf-8")) + [EOS_ID]


def decode_ids(ids) -> str:
    """token 序列 -> 文本，遇 EOS 停止，跳过 PAD/BOS。"""
    out = bytearray()
    for i in ids:
        i = int(i)
        if i == EOS_ID:
            break
        if i in (PAD_ID, BOS_ID):
            continue
        out.append(i)
    return out.decode("utf-8", errors="replace")
