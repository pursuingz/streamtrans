"""勘探 Qwen3.5-2B/9B 真实结构，输出报告到 docs/。
用法: python scripts/run_inspect.py --model Qwen/Qwen3.5-2B
"""
import argparse
import json
from pathlib import Path
from streamtrans.prune.inspect_model import load_meta_model, param_breakdown, dump_module_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--out", default="docs/model_inspection")
    args = ap.parse_args()

    model, cfg, loader = load_meta_model(args.model)
    # 覆盖已知组件：嵌入、各层、视觉塔、MTP 头。真实 module 名以 dump 为准。
    prefixes = ["embed_tokens", "lm_head", "layers", "visual", "vision", "mtp", "mlp", "self_attn"]
    bd = param_breakdown(model, group_prefixes=prefixes)
    names = dump_module_names(model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "_")
    (out_dir / f"{safe}.json").write_text(
        json.dumps({
            "model": args.model,
            "loader_class": loader,
            "config": cfg.to_dict(),
            "param_breakdown": bd,
            "module_names": names,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"loaded via: {loader}")
    print(f"total params: {bd['__total__'] / 1e9:.3f}B")
    print(f"embed_tokens: {bd['embed_tokens'] / 1e6:.1f}M")
    print(f"report -> {out_dir / (safe + '.json')}")


if __name__ == "__main__":
    main()
