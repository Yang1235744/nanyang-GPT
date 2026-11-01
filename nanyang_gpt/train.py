"""训练入口（在项目根目录运行）。

  python -m nanyang_gpt.train --config configs/nanyang_gpt_medium.yaml
"""

import argparse
import glob
import os
import random
import re

import torch
import yaml
from tqdm import tqdm

from nanyang_gpt.model import NanyangGPT
from nanyang_gpt.tokenizer import encode_stream
from nanyang_gpt.vision import TinyVisionEncoder, load_image

IMG_RE = re.compile(r"用户：\{img:([^}]+)\}([^\n]*)\n助手：[^\n]*")


def load_corpus(corpus_dir):
    """读语料；返回 (纯文本流字符串, 看图样本列表)。图片放 corpus_dir/images/。"""
    texts, img_entries = [], []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for m in IMG_RE.finditer(text):
            img_path = os.path.join(corpus_dir, "images", m.group(1))
            if os.path.isfile(img_path):
                img_entries.append((img_path, m.group(0)))   # 整段对话（不含标记）
        texts.append(IMG_RE.sub(
            lambda m: f"用户：{m.group(2)}\n助手：" + m.group(0).split("助手：", 1)[1], text))
    if not texts and not img_entries:
        raise RuntimeError(f"{corpus_dir} 里没有 .txt 语料")
    return texts, img_entries


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
    vcfg = cfg.get("vision")
    vis = TinyVisionEncoder(dim=cfg["gpt"]["dim"], **(vcfg or {})).to(device) if vcfg else None
    params = list(gpt.parameters()) + (list(vis.parameters()) if vis else [])
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t.get("weight_decay", 0.01))

    texts, img_entries = load_corpus(t["corpus_dir"])
    stream = []
    for text in texts:
        stream.extend(encode_stream(text))
    data = torch.tensor(stream, dtype=torch.long)
    n_img = vis.n_tokens if vis else 0
    print(f"[南阳GPT] 语料 {len(stream)} tokens, 看图样本 {len(img_entries)}, device={device}")

    start = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        gpt.load_state_dict(ckpt["gpt"])
        if vis and ckpt.get("vision") is not None:
            vis.load_state_dict(ckpt["vision"])
        start = ckpt.get("step", 0)
        print(f"已从 {args.resume} 恢复 (step={start})")

    os.makedirs(t["out_dir"], exist_ok=True)
    max_len = cfg["gpt"]["max_len"]
    p_img = t.get("p_img", 0.3) if (vis and img_entries) else 0.0
    pbar = tqdm(range(start, t["num_steps"]), initial=start, total=t["num_steps"])
    for step in pbar:
        if p_img > 0 and (len(data) <= max_len + 1 or random.random() < p_img):
            # 看图样本：图片在前缀，文本紧随其后（文本长度留出视觉 token 的位置）
            batch = [random.choice(img_entries) for _ in range(t["batch_size"])]
            ids_list = [encode_stream(dialog)[: max_len - n_img] for _, dialog in batch]
            L = max(len(s) for s in ids_list)
            x = torch.full((len(batch), L), 256, dtype=torch.long)   # PAD_ID
            for i, s in enumerate(ids_list):
                x[i, : len(s)] = torch.tensor(s)
            x = x.to(device)
            with torch.no_grad() if t.get("freeze_vision_input", False) else torch.enable_grad():
                imgs = torch.cat([load_image(p, vis.image_size) for p, _ in batch]).to(device)
            img_emb = vis(imgs)
            _, loss = gpt(x[:, :-1], x[:, 1:], img_emb=img_emb)
        else:
            ix = torch.randint(0, len(data) - max_len - 1, (t["batch_size"],))
            x = torch.stack([data[i:i + max_len] for i in ix]).to(device)
            y = torch.stack([data[i + 1:i + max_len + 1] for i in ix]).to(device)
            _, loss = gpt(x, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, t.get("grad_clip", 1.0))
        opt.step()

        if (step + 1) % t.get("log_every", 50) == 0:
            pbar.set_postfix_str(f"loss={loss.item():.4f}")
        if (step + 1) % t.get("save_every", 1000) == 0 or (step + 1) == t["num_steps"]:
            torch.save({"step": step + 1, "config": cfg, "gpt": gpt.state_dict(),
                        "vision": vis.state_dict() if vis else None},
                       os.path.join(t["out_dir"], "latest_gpt.pt"))

    print(f"训练完成，checkpoint 保存在 {t['out_dir']}/latest_gpt.pt")


if __name__ == "__main__":
    main()
