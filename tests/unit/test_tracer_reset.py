"""Reset modes of LLMTracer: chat prefill, thinking control, raw tokenization."""

import pytest


class TestChatPrefill:
    def test_trailing_assistant_message_is_continued(self, tracer):
        messages = [
            {"role": "user", "content": "Capital of France?"},
            {"role": "assistant", "content": "The capital is"},
        ]
        tracer.reset(messages=messages, mode="chat")
        # No generation prompt after the prefill; the rendered chat ends with
        # the assistant text so generation continues it.
        assert tracer.prompt.endswith("The capital is")

    def test_trailing_user_message_opens_assistant_turn(self, tracer):
        tracer.reset(
            messages=[{"role": "user", "content": "Hi"}], mode="chat"
        )
        assert tracer.prompt.endswith("assistant: ")

    def test_prefill_passes_continue_final_message(self, tracer, monkeypatch):
        captured = {}
        original = tracer.tokenizer.apply_chat_template

        def spy(messages, **kwargs):
            captured.update(kwargs)
            return original(messages, **kwargs)

        monkeypatch.setattr(tracer.tokenizer, "apply_chat_template", spy)
        tracer.reset(
            messages=[{"role": "assistant", "content": "Once upon"}], mode="chat"
        )
        assert captured["continue_final_message"] is True
        assert captured["add_generation_prompt"] is False

    def test_no_prefill_keeps_generation_prompt(self, tracer, monkeypatch):
        captured = {}
        original = tracer.tokenizer.apply_chat_template

        def spy(messages, **kwargs):
            captured.update(kwargs)
            return original(messages, **kwargs)

        monkeypatch.setattr(tracer.tokenizer, "apply_chat_template", spy)
        tracer.reset(messages=[{"role": "user", "content": "Hi"}], mode="chat")
        assert captured["continue_final_message"] is False
        assert captured["add_generation_prompt"] is True


class TestThinkingControl:
    def _spy_template(self, tracer, monkeypatch):
        captured = {}
        original = tracer.tokenizer.apply_chat_template

        def spy(messages, **kwargs):
            captured.update(kwargs)
            return original(messages, **kwargs)

        monkeypatch.setattr(tracer.tokenizer, "apply_chat_template", spy)
        return captured

    def test_off_uses_template_switch(self, tracer, monkeypatch):
        captured = self._spy_template(tracer, monkeypatch)
        tracer.reset(
            messages=[{"role": "user", "content": "Hi"}],
            mode="chat",
            thinking="off",
        )
        assert captured["enable_thinking"] is False
        assert captured["add_generation_prompt"] is True

    def test_prefill_opens_unclosed_think(self, tracer, monkeypatch):
        captured = self._spy_template(tracer, monkeypatch)
        tracer.reset(
            messages=[{"role": "user", "content": "Hi"}],
            mode="chat",
            thinking="prefill",
            think_prefill="Okay, the user wants",
        )
        assert tracer.prompt.endswith("<think>\nOkay, the user wants")
        assert captured["add_generation_prompt"] is True

    def test_conflicts_with_assistant_prefill(self, tracer):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Sure"},
        ]
        with pytest.raises(ValueError, match="assistant-prefill"):
            tracer.reset(messages=messages, mode="chat", thinking="off")

    def test_auto_is_unchanged(self, tracer, monkeypatch):
        captured = self._spy_template(tracer, monkeypatch)
        tracer.reset(messages=[{"role": "user", "content": "Hi"}], mode="chat")
        assert "enable_thinking" not in captured
        assert tracer.prompt.endswith("assistant: ")


class TestRawMode:
    def test_raw_mode_disables_auto_special_tokens(self, tracer, monkeypatch):
        captured = {}
        original = tracer.tokenizer.encode

        def spy(text, **kwargs):
            captured.update(kwargs)
            return original(text, **kwargs)

        monkeypatch.setattr(tracer.tokenizer, "encode", spy)
        tracer.reset(prompt="hello", mode="raw")
        assert captured["add_special_tokens"] is False
        assert tracer.mode == "raw"

    def test_completion_mode_keeps_special_tokens(self, tracer, monkeypatch):
        captured = {}
        original = tracer.tokenizer.encode

        def spy(text, **kwargs):
            captured.update(kwargs)
            return original(text, **kwargs)

        monkeypatch.setattr(tracer.tokenizer, "encode", spy)
        tracer.reset(prompt="hello", mode="completion")
        assert captured["add_special_tokens"] is True

    def test_raw_mode_still_parses_special_tokens_in_text(self, tracer):
        # add_special_tokens=False only stops the tokenizer from ADDING
        # bos/eos; specials typed in the text must still become single tokens.
        eos_id = tracer.tokenizer.eos_token_id
        tracer.reset(prompt="<|eos|>hi", mode="raw")
        ids = tracer.input_ids[0].tolist()
        assert ids[0] == eos_id
        assert len(ids) == 1 + len(
            tracer.tokenizer.encode("hi", add_special_tokens=False)
        )

    def test_raw_mode_generates(self, tracer):
        tracer.reset(prompt="Hello", mode="raw")
        tracer.step()
        assert len(tracer.history) == 1
