"""EOS handling: scalar, list, and generation_config-sourced EOS ids.

Regression for the old scalar-only comparison (`token_id == eos_token_id`)
which never matched when a model declares multiple EOS tokens (as Qwen chat
models do) and ignored generation_config entirely.
"""

from miru_tracer.core.tracer import LLMTracer, _collect_eos_ids


class TestCollectEosIds:
    def test_scalar_from_tokenizer(self, tiny_model, tiny_tokenizer):
        tracer = LLMTracer(tiny_model, tiny_tokenizer, device="cpu")
        assert tiny_tokenizer.eos_token_id in tracer.eos_ids

    def test_list_from_generation_config(self, tiny_model, tiny_tokenizer):
        tiny_model.generation_config.eos_token_id = [258, 259]
        try:
            tracer = LLMTracer(tiny_model, tiny_tokenizer, device="cpu")
            assert 258 in tracer.eos_ids
            assert 259 in tracer.eos_ids
            assert tracer.is_eos(258)
            assert tracer.is_eos(259)
            # Tokenizer EOS is still honored alongside
            assert tracer.is_eos(tiny_tokenizer.eos_token_id)
        finally:
            tiny_model.generation_config.eos_token_id = None

    def test_none_sources(self):
        class Empty:
            eos_token_id = None
            generation_config = None

        assert _collect_eos_ids(Empty(), Empty()) == frozenset()


class TestGenerationStopsAtEos:
    def test_stops_on_generation_config_eos(self, tiny_model, tiny_tokenizer):
        """Set the token greedy decoding will pick as a generation_config EOS
        (list form) and verify generation stops after it."""
        probe = LLMTracer(tiny_model, tiny_tokenizer, device="cpu")
        probe.reset("Hello")
        next_token = probe.peek(top_k=1).top_k_tokens[0]

        tiny_model.generation_config.eos_token_id = [next_token]
        try:
            tracer = LLMTracer(tiny_model, tiny_tokenizer, device="cpu")
            tracer.reset("Hello")
            steps = list(tracer.generate_stream(max_new_tokens=10))
            assert len(steps) == 1
            assert steps[0].token_id == next_token
        finally:
            tiny_model.generation_config.eos_token_id = None

    def test_stop_at_eos_false_continues(self, tiny_model, tiny_tokenizer):
        probe = LLMTracer(tiny_model, tiny_tokenizer, device="cpu")
        probe.reset("Hello")
        next_token = probe.peek(top_k=1).top_k_tokens[0]

        tiny_model.generation_config.eos_token_id = [next_token]
        try:
            tracer = LLMTracer(tiny_model, tiny_tokenizer, device="cpu")
            tracer.reset("Hello")
            steps = list(tracer.generate_stream(max_new_tokens=5, stop_at_eos=False))
            assert len(steps) == 5
        finally:
            tiny_model.generation_config.eos_token_id = None


class TestStopRequest:
    def test_request_stop_halts_generation(self, tracer):
        tracer.reset("Hello")
        produced = []
        for step in tracer.generate_stream(max_new_tokens=10):
            produced.append(step)
            if len(produced) == 2:
                tracer.request_stop()
        assert len(produced) == 2

    def test_stop_flag_cleared_on_new_stream(self, tracer):
        tracer.reset("Hello")
        tracer.request_stop()
        steps = list(tracer.generate_stream(max_new_tokens=3))
        assert len(steps) == 3
