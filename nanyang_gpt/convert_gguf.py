"""把南阳 GPT checkpoint 转成 GGUF（GPT-2 架构家族），供 Ollama / llama.cpp 使用。

南阳 GPT 的拓扑与 GPT-2 一致：可学习位置嵌入、pre-norm、qkv 合一投影、
GELU MLP、输入输出嵌入共享 —— 因此可以直接写成 gpt2 架构的 GGUF。
tokenizer 是我们的字节级词表（0-255 = UTF-8 字节），用 GPT-2 的
bytes-to-unicode 字母表落盘，llama.cpp 的 BPE 编解码天然兼容。

用法：
  pip install gguf
  python -m nanyang_gpt.convert_gguf --ckpt runs/nanyang_gpt/latest_gpt.pt \
      --out nanyang-gpt-f16.gguf --ftype f16

然后接入 Ollama：
  ollama create nanyang-gpt -f Modelfile
  ollama run nanyang-gpt
"""

import argparse

import numpy as np
import torch

from nanyang_gpt.model import NanyangGPT
from nanyang_gpt.tokenizer import PAD_ID, BOS_ID, EOS_ID, VOCAB_SIZE


def gpt2_byte_alphabet():
    """GPT-2 的 bytes_to_unicode：可打印字节映射自身，其余映射到 256+n。"""
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def remap_state_dict(sd):
    """南阳 GPT state_dict -> GGUF gpt2 张量名。Conv1D 布局要求线性层转置。"""
    out = {
        "token_embd.weight": sd["tok.weight"],
        "position_embd.weight": sd["pos.weight"],
        "output_norm.weight": sd["ln.weight"],
        "output_norm.bias": sd["ln.bias"],
    }
    n_blk = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".n1.weight"))
    for i in range(n_blk):
        p = f"blocks.{i}."
        out[f"blk.{i}.attn_norm.weight"] = sd[p + "n1.weight"]
        out[f"blk.{i}.attn_norm.bias"] = sd[p + "n1.bias"]
        out[f"blk.{i}.attn_qkv.weight"] = sd[p + "qkv.weight"].T.contiguous()
        out[f"blk.{i}.attn_qkv.bias"] = sd[p + "qkv.bias"]
        out[f"blk.{i}.attn_output.weight"] = sd[p + "proj.weight"].T.contiguous()
        out[f"blk.{i}.attn_output.bias"] = sd[p + "proj.bias"]
        out[f"blk.{i}.ffn_norm.weight"] = sd[p + "n2.weight"]
        out[f"blk.{i}.ffn_norm.bias"] = sd[p + "n2.bias"]
        out[f"blk.{i}.ffn_up.weight"] = sd[p + "mlp.0.weight"].T.contiguous()
        out[f"blk.{i}.ffn_up.bias"] = sd[p + "mlp.0.bias"]
        out[f"blk.{i}.ffn_down.weight"] = sd[p + "mlp.2.weight"].T.contiguous()
        out[f"blk.{i}.ffn_down.bias"] = sd[p + "mlp.2.bias"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="latest_gpt.pt 路径")
    ap.add_argument("--out", default="nanyang-gpt-f16.gguf")
    ap.add_argument("--ftype", choices=["f32", "f16"], default="f16")
    args = ap.parse_args()

    import gguf

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    gcfg = ckpt["config"]["gpt"]
    sd = ckpt["gpt"]
    # 丢弃共享权重的重复引用，避免写两份
    sd = {k: v for k, v in sd.items() if k != "head.weight"}

    dim = gcfg["dim"]
    tensors = remap_state_dict(sd)
    ftype = gguf.LlamaFileType.MOSTLY_F16 if args.ftype == "f16" else gguf.LlamaFileType.ALL_F32

    alphabet = gpt2_byte_alphabet()
    tokens = [alphabet[i] for i in range(256)] + ["<|pad|>", "<|bos|>", "<|eos|>"]
    assert len(tokens) == VOCAB_SIZE

    writer = gguf.GGUFWriter(args.out, arch="gpt2")
    writer.add_name("nanyang-GPT")
    writer.add_description("从零自研的字节级对话大模型（南阳 GPT），工具调用 <|gen_image|> / <|gen_video|>")
    writer.add_context_length(gcfg["max_len"])
    writer.add_embedding_length(dim)
    writer.add_block_count(gcfg["depth"])
    writer.add_head_count(gcfg["heads"])
    writer.add_feed_forward_length(dim * 4)
    writer.add_layer_norm_eps(1e-5)
    writer.add_file_type(ftype)
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_types([1] * 256 + [3, 3, 3])          # 字节=NORMAL，特殊符号=CONTROL
    writer.add_bos_token_id(BOS_ID)
    writer.add_eos_token_id(EOS_ID)
    writer.add_pad_token_id(PAD_ID)
    writer.add_unk_token_id(PAD_ID)
    writer.add_token_merges([])                            # 字节级，无合并规则

    for name, t in tensors.items():
        writer.add_tensor(name, t.detach().to(torch.float32).numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    n_params = sum(v.numel() for v in tensors.values())
    print(f"GGUF 写出: {args.out}  ({args.ftype}, {n_params / 1e6:.1f}M 参数, "
          f"{gcfg['depth']} 层 x dim {gcfg['dim']}, vocab {len(tokens)})")
    print("下一步: ollama create nanyang-gpt -f Modelfile && ollama run nanyang-gpt")


if __name__ == "__main__":
    main()
