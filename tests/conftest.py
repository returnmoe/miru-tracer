"""Shared fixtures: a tiny offline model for unit tests, Qwen3-0.6B for integration.

The tiny model is a randomly initialized 2-layer Llama built from a config —
no network, instantiates in under a second, and uses the same modern cache /
GQA / RoPE machinery as Qwen3, so it exercises the exact transformers code
paths the real model does.
"""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

TINY_VOCAB_SIZE = 260  # 256 byte tokens + eos/pad + 2 spare ids for EOS-list tests

SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def build_byte_tokenizer() -> PreTrainedTokenizerFast:
    """A byte-level BPE tokenizer over the 256-byte alphabet, built in code."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.add_special_tokens({"eos_token": "<|eos|>", "pad_token": "<|pad|>"})
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    return tokenizer


def build_tiny_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=TINY_VOCAB_SIZE,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval()
    # A random model has no meaningful EOS; unit tests set one explicitly
    # when they need EOS behavior.
    model.generation_config.eos_token_id = None
    return model


@pytest.fixture(scope="session")
def tiny_model():
    return build_tiny_model()


@pytest.fixture(scope="session")
def tiny_tokenizer():
    return build_byte_tokenizer()


@pytest.fixture()
def tracer(tiny_model, tiny_tokenizer):
    from miru_tracer.core.tracer import LLMTracer

    return LLMTracer(tiny_model, tiny_tokenizer, device="cpu", seed=0)


@pytest.fixture(scope="session")
def qwen3():
    """Qwen/Qwen3-0.6B for integration tests. Skips if unavailable."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen3-0.6B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - network-dependent
        pytest.skip(f"{name} unavailable: {e}")
    return model, tokenizer
