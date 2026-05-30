import pytest

from streamtrans.distill.prompt import assemble_labels, direction_tag, render_prompt


def test_assemble_labels_masks_prompt_only():
    inp, lab = assemble_labels([5, 6, 7], [10, 11])
    assert inp == [5, 6, 7, 10, 11]
    assert lab == [-100, -100, -100, 10, 11]   # 仅 target 段计 loss


def test_render_prompt_has_direction_tag():
    p = render_prompt("你好", "zh2en")
    assert "[zh2en]" in p and "你好" in p


def test_direction_tag_validates():
    assert direction_tag("en2zh") == "[en2zh]"
    with pytest.raises(ValueError):
        direction_tag("zh2fr")
