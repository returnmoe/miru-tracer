"""Model Loader UI helpers."""

from miru_tracer.ui.model_loader import (
    CUSTOM_MODEL_CHOICE,
    QUICK_MODEL_CHOICES,
    resolve_model_name,
    toggle_custom_model_field,
)


def test_quick_model_choices_start_with_qwen3_defaults():
    assert QUICK_MODEL_CHOICES[:2] == (
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-4B",
    )
    assert QUICK_MODEL_CHOICES[-1] == CUSTOM_MODEL_CHOICE


def test_resolve_model_name_uses_preset_unless_other():
    assert resolve_model_name("Qwen/Qwen3-4B", "ignored/model") == "Qwen/Qwen3-4B"
    assert resolve_model_name(CUSTOM_MODEL_CHOICE, " custom/model ") == "custom/model"
    assert resolve_model_name(CUSTOM_MODEL_CHOICE, None) == ""


def test_toggle_custom_model_field_visibility():
    shown = toggle_custom_model_field(CUSTOM_MODEL_CHOICE)
    assert shown["visible"] is True
    assert shown["value"] == ""

    hidden = toggle_custom_model_field("Qwen/Qwen3-4B")
    assert hidden["visible"] is False
    assert hidden["value"] == "Qwen/Qwen3-4B"
