"""在中英测试集上评学生翻译质量(BLEU/chrF)。

学生在 new-id 空间：prompt 经 VocabRemapper 编码 → generate → 解码回文本。
用法:
  python scripts/run_eval.py --ckpt checkpoints/distilled_20260530 \
         --test data/test_zh_en.jsonl --base Qwen/Qwen3.5-2B \
         --vocab-map checkpoints/pruned_20260530/vocab_map.json
"""
import argparse
import json
from pathlib import Path

import torch

from streamtrans.data.vocab_remap import VocabRemapper
from streamtrans.distill.prompt import render_prompt
from streamtrans.eval.quality import corpus_bleu, corpus_chrf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="学生 checkpoint")
    ap.add_argument("--test", required=True, help="测试 jsonl: {src,tgt,direction}")
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B", help="取 tokenizer")
    ap.add_argument("--vocab-map", required=True)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    remapper = VocabRemapper.from_file(args.vocab_map)
    eos_new = remapper.old2new.get(int(tok.eos_token_id))
    model = Qwen3_5ForCausalLM.from_pretrained(args.ckpt, trust_remote_code=True).to(dev).eval()

    rows = [json.loads(l) for l in Path(args.test).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    hyps, refs = [], []
    for r in rows:
        p_new = remapper.encode(tok, render_prompt(r["src"], r["direction"]))
        ids = torch.tensor([p_new], device=dev)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=args.max_new, do_sample=False,
                                 eos_token_id=eos_new, pad_token_id=eos_new)
        out_new = gen[0, ids.shape[1]:].tolist()
        hyps.append(remapper.decode(tok, out_new, skip_special_tokens=True))
        refs.append(r["tgt"])

    print(f"samples={len(hyps)}  BLEU={corpus_bleu(hyps, refs):.2f}  chrF={corpus_chrf(hyps, refs):.2f}")


if __name__ == "__main__":
    main()
