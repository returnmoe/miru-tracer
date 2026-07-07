"""End-to-end integration with Qwen/Qwen3-0.6B on CPU.

Opt in with: pytest -m integration
Downloads ~1.4GB on first run (cached in HF_HOME afterwards).
"""

import json

import pytest

from miru_tracer.core.sampling import SamplingParams
from miru_tracer.core.schema import parse_log
from miru_tracer.core.tracer import LLMTracer
from miru_tracer.visualization.plots import (
    get_generation_stats,
    plot_probability_visualizations,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def qwen_tracer(qwen3):
    model, tokenizer = qwen3
    return LLMTracer(model, tokenizer, device="cpu", seed=0)


class TestQwen3EndToEnd:
    def test_eos_ids_include_qwen_chat_terminators(self, qwen_tracer):
        """Qwen3 declares <|im_end|> and <|endoftext|> — both must count as EOS."""
        tokenizer = qwen_tracer.tokenizer
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        assert im_end in qwen_tracer.eos_ids
        assert endoftext in qwen_tracer.eos_ids

    def test_full_interactive_loop(self, qwen_tracer):
        """step x3 -> peek purity -> undo -> continue -> export -> re-import."""
        tracer = qwen_tracer
        tracer.reset("The capital of France is")

        # Step 3 tokens greedily
        steps = [tracer.step() for _ in range(3)]
        assert len(tracer.history) == 3
        assert all(0.0 <= step.probability <= 1.0 for step in steps)

        # Peek with different display params — cheap, state untouched
        dist_a = tracer.peek(top_k=5, temperature=0.5)
        dist_b = tracer.peek(top_k=15, temperature=2.0)
        assert dist_a.top_k_tokens[0] == dist_b.top_k_tokens[0]  # same argmax
        assert len(tracer.history) == 3

        # Undo two, regenerate greedily — identical tokens (cache correctness)
        replay_expected = [steps[1].token_id, steps[2].token_id]
        assert tracer.undo(2)
        replayed = [tracer.step().token_id for _ in range(2)]
        assert replayed == replay_expected

        # Continue via stream
        more = list(
            tracer.generate_stream(max_new_tokens=3, params=SamplingParams())
        )
        assert 1 <= len(more) <= 3

        # Export -> parse round trip
        params = SamplingParams()
        exported = json.loads(json.dumps(tracer.export_to_dict(params)))
        log = parse_log(exported)
        assert log.schema_version == 2
        assert log.num_steps == len(tracer.history)
        assert "France" in log.full_text

        # Plots build from the real history
        figures = plot_probability_visualizations(log.history, top_k=5)
        assert len(figures) == 2
        stats = get_generation_stats(log.history)
        assert stats["total_steps"] == len(tracer.history)

    def test_greedy_answer_is_sane(self, qwen_tracer):
        """A 0.6B model should still complete 'The capital of France is' -> Paris."""
        tracer = qwen_tracer
        tracer.reset("The capital of France is")
        text = tracer.generate(max_new_tokens=5)
        assert "Paris" in text

    def test_chat_template_mode(self, qwen_tracer):
        tracer = qwen_tracer
        tracer.reset(
            messages=[{"role": "user", "content": "Say hi"}],
            mode="chat",
        )
        assert tracer.has_chat_template
        assert "<|im_start|>" in tracer.prompt
        step = tracer.step()
        assert step.token_id is not None
