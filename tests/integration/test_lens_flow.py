"""End-to-end lens + interventions flow through the real Gradio app.

Uses the tiny offline model (patched into the ModelManager singleton) with a
lens fitted on the fly; in the integration folder because it launches a real
server and drives it with gradio_client.
"""

import pytest
from gradio_client import Client

from miru_tracer.core._jlens import fit, from_hf

pytestmark = pytest.mark.integration

PORT = 7871


@pytest.fixture()
def lens_app(tiny_model, tiny_tokenizer, tmp_path, monkeypatch):
    from miru_tracer.core import lens as lens_module
    from miru_tracer.core.lens import LensStore
    from miru_tracer.core.model_manager import ModelManager
    from miru_tracer.ui.lens_common import set_active_interventions

    monkeypatch.setattr(ModelManager, "_model", tiny_model)
    monkeypatch.setattr(ModelManager, "_tokenizer", tiny_tokenizer)
    monkeypatch.setattr(ModelManager, "_device", "cpu")
    monkeypatch.setattr(ModelManager, "_model_name", "tiny/test-model")

    store = LensStore(base_dir=tmp_path)
    monkeypatch.setattr(lens_module, "_lens_store", store)
    lens = fit(
        from_hf(tiny_model, tiny_tokenizer, force_bos=False),
        [
            "Hello world, this is a much longer test prompt for fitting the lens today.",
            "The quick brown fox jumps over the lazy dog again and again without stop.",
        ],
        dim_batch=8,
    )
    path = store.lens_path("tiny/test-model")
    path.parent.mkdir(parents=True, exist_ok=True)
    lens.save(str(path))

    set_active_interventions([])
    from miru_tracer.app import create_app

    app = create_app()
    app.queue()
    app.launch(
        server_name="127.0.0.1", server_port=PORT, prevent_thread_lock=True, quiet=True
    )
    try:
        yield Client(f"http://127.0.0.1:{PORT}/", verbose=False)
    finally:
        app.close()
        set_active_interventions([])


class TestLensTabFlow:
    def test_multi_intervention_steering_shows_in_text_and_readouts(self, lens_app):
        client = lens_app
        # Two simultaneous interventions — beyond Neuronpedia's single one.
        client.predict("steer", "A", "", 0, 3.0, "jacobian", api_name="/add_intervention")
        result = client.predict(
            "steer", "B", "", 1, 2.0, "logit", api_name="/add_intervention"
        )
        assert "2 intervention(s)" in result[1]

        out = client.predict(
            "Completion", "Hello world", "[]", 6, "greedy", 1.0,
            "Jacobian", 0, -1, 1, 8, False, "A",
            api_name="/generate_and_analyze",
        )
        readout_df, dist, heatmap, pinned, _tokens, status, text = out[:7]
        assert "Interventions active: 2" in status
        # Both steers compose; the final-layer one dominates greedy decoding,
        # so the generated text must consist of steered tokens.
        generated = text[len("Hello world"):]
        assert generated and set(generated) <= {"A", "B"}
        # ...and both steered tokens surface prominently in the readouts.
        top_tokens = [row[0] for row in readout_df["data"][:5]]
        assert "A" in top_tokens and "B" in top_tokens
        assert heatmap is not None and dist is not None and pinned is not None

    def test_remove_and_clear_interventions(self, lens_app):
        client = lens_app
        client.predict("ablate", "C", "", 0, 1.0, "logit", api_name="/add_intervention")
        removed = client.predict(0, api_name="/remove_intervention")
        assert "Removed #0" in removed[1]
        cleared = client.predict(api_name="/clear_interventions")
        assert "Cleared" in cleared[1]

    def test_logit_mode_without_fitted_lens_still_works(self, lens_app):
        out = lens_app.predict(
            "Completion", "Hello world", "[]", 3, "greedy", 1.0,
            "Logit", 0, -1, 1, 5, False, "",
            api_name="/generate_and_analyze",
        )
        status = out[5]
        assert "logit lens" in status
        assert out[2] is not None  # heatmap
