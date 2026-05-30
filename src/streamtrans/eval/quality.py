"""翻译质量指标：BLEU / chrF（sacrebleu），COMET 可选（重依赖）。"""
from typing import List, Optional


def corpus_bleu(hyps: List[str], refs: List[str]) -> float:
    import sacrebleu
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def corpus_chrf(hyps: List[str], refs: List[str]) -> float:
    import sacrebleu
    return sacrebleu.corpus_chrf(hyps, [refs]).score


def corpus_comet(srcs: List[str], hyps: List[str], refs: List[str]) -> Optional[float]:
    """COMET 需 unbabel-comet（重依赖）。未安装则返回 None。"""
    try:
        from comet import download_model, load_from_checkpoint
    except Exception:  # noqa: BLE001
        return None
    path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(path)
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    return float(model.predict(data, progress_bar=False).system_score)
