"""训练入口（在项目根目录运行）。

  python -m nanyang_gpt.train --config configs/nanyang_gpt_medium.yaml
"""

import argparse
import glob
import os
import random

import torch
import yaml
from tqdm import tqdm

from nanyang_gpt.model import NanyangGPT
from nanyang_gpt.tokenizer import encode_stream


def load_corpus(corpus_dir):
    texts = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
    if not texts:
        raise RuntimeError(f"{corpus_dir} 里没有 .txt 语料")
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    t = cfg["train"]
    torch.manual_seed(cfg.get("seed", 42))
    random.seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gpt = NanyangGPT(**cfg["gpt"]).to(device)
    opt = torch.optim.AdamW(gpt.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.01))

    stream = []
    for text in load_corpus(t["corpus_dir"]):
        stream.extend(encode_stream(text))
    data = torch.tensor(stream, dtype=torch.long)
    print(f"[南阳GPT] 语料 {len(stream)} tokens, device={device}")

    start = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        gpt.load_state_dict(ckpt["gpt"])
        start = ckpt.get("step", 0)
        print(f"已从 {args.resume} 恢复 (step={start})")

    os.makedirs(t["out_dir"], exist_ok=True)
    max_len = cfg["gpt"]["max_len"]
    pbar = tqdm(range(start, t["num_steps"]), initial=start, total=t["num_steps"])
    for step in pbar:
        ix = torch.randint(0, len(data) - max_len - 1, (t["batch_size"],))
        x = torch.stack([data[i:i + max_len] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + max_len + 1] for i in ix]).to(device)
        _, loss = gpt(x, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), t.get("grad_clip", 1.0))
        opt.step()

        if (step + 1) % t.get("log_every", 50) == 0:
            pbar.set_postfix_str(f"loss={loss.item():.4f}")
        if (step + 1) % t.get("save_every", 1000) == 0 or (step + 1) == t["num_steps"]:
            torch.save({"step": step + 1, "config": cfg, "gpt": gpt.state_dict()},
                       os.path.join(t["out_dir"], "latest_gpt.pt"))

    print(f"训练完成，checkpoint 保存在 {t['out_dir']}/latest_gpt.pt")


if __name__ == "__main__":
    main()
