"""Environment parsing — regression for the MIRU_DEBUG == ("1" or "true") bug."""

import pytest

from miru_tracer.config import Settings, env_bool, env_int, env_str


class TestEnvBool:
    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "on", " true "])
    def test_truthy(self, value, monkeypatch):
        monkeypatch.setenv("X", value)
        assert env_bool("X") is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "2", "enabled"])
    def test_falsy(self, value, monkeypatch):
        monkeypatch.setenv("X", value)
        assert env_bool("X") is False

    def test_unset_uses_default(self, monkeypatch):
        monkeypatch.delenv("X", raising=False)
        assert env_bool("X") is False
        assert env_bool("X", default=True) is True


class TestEnvInt:
    def test_parses(self, monkeypatch):
        monkeypatch.setenv("X", "7861")
        assert env_int("X", 7860) == 7861

    @pytest.mark.parametrize("value", ["", "  ", "abc"])
    def test_invalid_uses_default(self, value, monkeypatch):
        monkeypatch.setenv("X", value)
        assert env_int("X", 7860) == 7860


class TestSettings:
    def test_defaults(self, monkeypatch):
        for var in (
            "MIRU_DEBUG",
            "MIRU_SERVER_NAME",
            "MIRU_SERVER_PORT",
            "GRADIO_SERVER_NAME",
            "GRADIO_SERVER_PORT",
            "MIRU_ALLOW_REMOTE_CODE",
            "MIRU_AUTH_USERNAME",
            "MIRU_AUTH_PASSWORD",
            "MIRU_MAX_NEW_TOKENS",
            "MIRU_MAX_LOG_TOP_K",
            "MIRU_MAX_FULL_PROB_STEPS",
            "MIRU_MAX_LENS_CELLS",
        ):
            monkeypatch.delenv(var, raising=False)
        settings = Settings.from_env()
        assert settings.debug is False
        assert settings.server_name == "127.0.0.1"
        assert settings.server_port == 7860
        assert settings.allow_remote_code is False
        assert settings.auth is None
        assert settings.max_new_tokens == 1000
        assert settings.max_log_top_k == 256
        assert settings.max_full_prob_steps == 128
        assert settings.max_lens_cells == 8192

    def test_miru_debug_true_string(self, monkeypatch):
        """The exact case the old code got wrong."""
        monkeypatch.setenv("MIRU_DEBUG", "true")
        assert Settings.from_env().debug is True

    def test_miru_vars_win_over_gradio_vars(self, monkeypatch):
        monkeypatch.setenv("MIRU_SERVER_NAME", "0.0.0.0")
        monkeypatch.setenv("GRADIO_SERVER_NAME", "10.0.0.1")
        monkeypatch.setenv("MIRU_SERVER_PORT", "8000")
        monkeypatch.setenv("GRADIO_SERVER_PORT", "9000")
        settings = Settings.from_env()
        assert settings.server_name == "0.0.0.0"
        assert settings.server_port == 8000

    def test_gradio_vars_as_fallback(self, monkeypatch):
        monkeypatch.delenv("MIRU_SERVER_NAME", raising=False)
        monkeypatch.delenv("MIRU_SERVER_PORT", raising=False)
        monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")
        monkeypatch.setenv("GRADIO_SERVER_PORT", "9000")
        settings = Settings.from_env()
        assert settings.server_name == "0.0.0.0"
        assert settings.server_port == 9000

    def test_paired_auth(self, monkeypatch):
        monkeypatch.setenv("MIRU_AUTH_USERNAME", "miru")
        monkeypatch.setenv("MIRU_AUTH_PASSWORD", "secret")
        assert Settings.from_env().auth == ("miru", "secret")

    def test_partial_auth_rejected(self, monkeypatch):
        monkeypatch.setenv("MIRU_AUTH_USERNAME", "miru")
        monkeypatch.delenv("MIRU_AUTH_PASSWORD", raising=False)
        with pytest.raises(ValueError, match="must be set together"):
            Settings.from_env()

    def test_limits_are_configurable_and_positive(self, monkeypatch):
        monkeypatch.setenv("MIRU_MAX_NEW_TOKENS", "12")
        monkeypatch.setenv("MIRU_MAX_LOG_TOP_K", "0")
        settings = Settings.from_env()
        assert settings.max_new_tokens == 12
        assert settings.max_log_top_k == 1


class TestEnvStr:
    def test_fallback_order(self, monkeypatch):
        monkeypatch.delenv("A", raising=False)
        monkeypatch.setenv("B", "from-b")
        assert env_str("A", "default", "B") == "from-b"
        monkeypatch.delenv("B", raising=False)
        assert env_str("A", "default", "B") == "default"
