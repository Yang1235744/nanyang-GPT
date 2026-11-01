"""命令行聊天（在项目根目录运行）。

  python -m nanyang_gpt.chat --ckpt runs/nanyang_gpt/latest_gpt.pt

模型输出里若出现 <|gen_image|> / <|gen_video|> 标记，会被解析为工具调用；
本仓库只负责识别并提示，真正生成图片/视频由 nanyang-Helios1.5 执行。
"""

import argparse

import torch

from nanyang_gpt.model import BOT, USER, NanyangGPT, parse_tool
from nanyang_gpt.tokenizer import BOS_ID, decode_ids, encode_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", default=None, help="可选：让模型看一张图再聊")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    gpt = NanyangGPT(**ckpt["config"]["gpt"]).to(device)
    gpt.load_state_dict(ckpt["gpt"])

    img_emb = None
    if args.image:
        if "vision" not in ckpt:
            raise SystemExit("这个 checkpoint 没有视觉模块，请用带 vision 的训练配置")
        from nanyang_gpt.vision import TinyVisionEncoder, load_image
        vis = TinyVisionEncoder(dim=ckpt["config"]["gpt"]["dim"],
                                **ckpt["config"].get("vision", {})).to(device)
        vis.load_state_dict(ckpt["vision"])
        vis.eval()
        with torch.no_grad():
            img_emb = vis(load_image(args.image, vis.image_size).to(device))
        print(f"[南阳GPT] 已看图: {args.image} -> {img_emb.shape[1]} 个视觉 token")
    print(f"[南阳GPT] 已加载 {args.ckpt} (step={ckpt.get('step', 0)})，输入 exit 退出")

    history = ""
    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        history += f"{USER}{q}\n{BOT}"
        n_img = img_emb.shape[1] if img_emb is not None else 0
        ids = torch.tensor([[BOS_ID] + list(history.encode("utf-8"))[-(gpt.max_len - n_img):]],
                           device=device)
        out = gpt.generate(ids, max_new=args.max_new,
                           temperature=args.temperature, top_k=args.top_k,
                           img_emb=img_emb)[0].tolist()
        reply = decode_ids(out[ids.shape[1]:]).split(f"\n{USER}")[0]   # 到下一轮用户发言前停
        text, tool, prompt = parse_tool(reply)
        if tool:
            print(f"南阳GPT: {text}\n[工具调用] {tool}: {prompt}  (交给 nanyang-Helios1.5 生成)")
        else:
            print(f"南阳GPT: {text}")
        history += reply + "\n"
        history = history[-2000:]                # 简单截断，防止上下文无限增长


if __name__ == "__main__":
    main()
