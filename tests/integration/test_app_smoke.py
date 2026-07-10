"""App-level smoke tests: the Blocks tree builds and serves.

These run with the tiny offline model (via a patched ModelManager), so they
are in the integration folder only because they spin up a real Gradio server.
"""


import socket

import httpx
import pytest

from miru_tracer import __version__
from miru_tracer.app import create_app
from miru_tracer.ui.theme import launch_kwargs

pytestmark = pytest.mark.integration


class TestAppSmoke:
    def test_create_app_builds(self):
        app = create_app()
        assert app is not None

    def test_app_serves_http(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        app = create_app()
        app.queue()
        try:
            app.launch(
                server_name="127.0.0.1",
                server_port=port,
                prevent_thread_lock=True,
                quiet=True,
                **launch_kwargs(__version__),
            )
            response = httpx.get(f"http://127.0.0.1:{port}/", timeout=10)
            assert response.status_code == 200
            assert "Miru Tracer" in response.text
        finally:
            app.close()

    def test_handlers_work_against_tiny_model(
        self, tiny_model, tiny_tokenizer, monkeypatch
    ):
        """Drive the interactive-mode flow at the session/tracer level the way
        the UI handlers do, with the ModelManager patched to the tiny model."""
        from miru_tracer.core.model_manager import ModelManager
        from miru_tracer.core.session_manager import get_session_manager

        manager = ModelManager()
        monkeypatch.setattr(ModelManager, "_model", tiny_model)
        monkeypatch.setattr(ModelManager, "_tokenizer", tiny_tokenizer)
        monkeypatch.setattr(ModelManager, "_device", "cpu")
        assert manager.is_loaded()

        session_manager = get_session_manager()
        session_id = session_manager.create_session(
            manager.get_model(), manager.get_tokenizer(), manager.get_device()
        )
        session = session_manager.get_session(session_id)
        with session.lock:
            session.tracer.reset("Hello")
            session.tracer.step()
            assert len(session.tracer.history) == 1
        session_manager.delete_session(session_id)
