"""SessionManager: isolation, disposal, expiry, and concurrency."""

import threading
from concurrent.futures import ThreadPoolExecutor

from miru_tracer.core.session_manager import SessionManager


class TestSessions:
    def test_create_and_get(self, tiny_model, tiny_tokenizer):
        manager = SessionManager()
        session_id = manager.create_session(tiny_model, tiny_tokenizer, "cpu")
        session = manager.get_session(session_id)
        assert session is not None
        assert session.tracer is not None
        assert isinstance(session.lock, type(threading.Lock()))

    def test_get_unknown_session_returns_none(self):
        manager = SessionManager()
        assert manager.get_session("nope") is None
        assert manager.get_tracer("nope") is None

    def test_single_session_enforced(self, tiny_model, tiny_tokenizer):
        manager = SessionManager()
        first = manager.create_session(tiny_model, tiny_tokenizer, "cpu")
        second = manager.create_session(tiny_model, tiny_tokenizer, "cpu")
        assert manager.get_session(first) is None
        assert manager.get_session(second) is not None
        assert manager.get_session_count() == 1

    def test_delete_session(self, tiny_model, tiny_tokenizer):
        manager = SessionManager()
        session_id = manager.create_session(tiny_model, tiny_tokenizer, "cpu")
        assert manager.delete_session(session_id) is True
        assert manager.get_session(session_id) is None
        assert manager.delete_session(session_id) is False

    def test_delete_releases_tracer_tensors(self, tiny_model, tiny_tokenizer):
        manager = SessionManager()
        session_id = manager.create_session(tiny_model, tiny_tokenizer, "cpu")
        session = manager.get_session(session_id)
        session.tracer.reset("Hello")
        session.tracer.step()
        assert session.tracer.input_ids is not None
        manager.delete_session(session_id)
        assert session.tracer.input_ids is None
        assert session.tracer._kv is None

    def test_cleanup_expired_sessions(self, tiny_model, tiny_tokenizer):
        manager = SessionManager(cleanup_timeout_minutes=0)
        session_id = manager.create_session(tiny_model, tiny_tokenizer, "cpu")
        # timeout of 0 minutes: any session is immediately stale
        assert manager.cleanup_old_sessions() == 1
        assert manager.get_session(session_id) is None

    def test_cleanup_keeps_fresh_sessions(self, tiny_model, tiny_tokenizer):
        manager = SessionManager(cleanup_timeout_minutes=30)
        session_id = manager.create_session(tiny_model, tiny_tokenizer, "cpu")
        assert manager.cleanup_old_sessions() == 0
        assert manager.get_session(session_id) is not None

    def test_session_info(self, tiny_model, tiny_tokenizer):
        manager = SessionManager()
        session_id = manager.create_session(tiny_model, tiny_tokenizer, "cpu")
        manager.get_session(session_id).tracer.reset("Hello")
        info = manager.get_session_info(session_id)
        assert info["steps"] == 0
        assert info["prompt"] == "Hello"
        assert manager.get_session_info("nope") is None

    def test_logging_and_interactive_sessions_coexist(self, tiny_model, tiny_tokenizer):
        manager = SessionManager()
        interactive = manager.create_session(
            tiny_model, tiny_tokenizer, "cpu", kind="interactive"
        )
        logging = manager.create_session(
            tiny_model, tiny_tokenizer, "cpu", kind="logging"
        )
        assert manager.get_session(interactive) is not None
        assert manager.get_session(logging) is not None

    def test_model_generation_invalidates_lookup(self, tiny_model, tiny_tokenizer):
        manager = SessionManager()
        session_id = manager.create_session(
            tiny_model, tiny_tokenizer, "cpu", model_generation=3
        )
        assert manager.get_session(session_id, expected_generation=3) is not None
        assert manager.get_session(session_id, expected_generation=4) is None


class TestConcurrency:
    def test_concurrent_create_get_delete(self, tiny_model, tiny_tokenizer):
        """Hammer the manager from many threads; no exceptions, no leaks."""
        manager = SessionManager()
        errors = []

        def worker(_):
            try:
                session_id = manager.create_session(tiny_model, tiny_tokenizer, "cpu")
                session = manager.get_session(session_id)
                if session is not None:  # may already be superseded by another thread
                    with session.lock:
                        session.tracer.reset("hi")
                        session.tracer.step()
                manager.delete_session(session_id)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(24)))

        assert errors == []
        assert manager.get_session_count() == 0

    def test_step_under_lock_while_deleting(self, tiny_model, tiny_tokenizer):
        """A deletion racing an in-flight step must not corrupt or raise."""
        manager = SessionManager()
        session_id = manager.create_session(tiny_model, tiny_tokenizer, "cpu")
        session = manager.get_session(session_id)
        session.tracer.reset("Hello")

        stop = threading.Event()
        errors = []

        def stepper():
            try:
                while not stop.is_set():
                    with session.lock:
                        if session.tracer.input_ids is None:
                            break
                        session.tracer.step()
            except Exception as e:  # pragma: no cover
                errors.append(e)

        thread = threading.Thread(target=stepper)
        thread.start()
        manager.delete_session(session_id)
        stop.set()
        thread.join(timeout=10)
        assert errors == []
