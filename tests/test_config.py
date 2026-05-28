import pytest
from pathlib import Path
from streamtrans.config import PruneConfig, load_config


def test_load_prune_config(tmp_path: Path):
    yaml_text = """
starter_model: Qwen/Qwen3.5-2B
target_params_b: 0.7
target_vocab_size: 110000
keep_languages: [zh, en]
reserve_languages: [ja, ko, fr, de, es, ru]
drop_vision: true
"""
    p = tmp_path / "prune.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(p, PruneConfig)
    assert cfg.starter_model == "Qwen/Qwen3.5-2B"
    assert cfg.target_params_b == 0.7
    assert cfg.target_vocab_size == 110000
    assert cfg.drop_vision is True
    assert "ja" in cfg.reserve_languages


def test_invalid_config_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("target_params_b: not_a_number\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_config(p, PruneConfig)
