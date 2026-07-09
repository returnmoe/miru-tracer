"""End-to-end lens + interventions flow through the real Gradio app.

Uses the tiny offline model (patched into the ModelManager singleton) with a
lens fitted on the fly; in the integration folder because it launches a real
server and drives it with gradio_client.
"""

import pytest
from gradio_client import Client

from miru_tracer.core._jlens import fit, from_hf
from miru_tracer.core.lens_io import save_lens

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
    save_lens(lens, path)

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
        client.predict(
            "steer", "A", "Text", "", "Text", "0-1", 2.0, "jacobian",
            api_name="/add_intervention",
        )
        result = client.predict(
            "steer", "B", "Text", "", "Text", 1, 2.0, "logit",
            api_name="/add_intervention",
        )
        assert "<table" in result[0]
        assert "steer" in result[0] and "jacobian" in result[0] and "logit" in result[0]
        assert result[0].count('data-miru-iv-row="') == 2
        assert ">0-1<" in result[0]
        assert "2 enabled intervention group(s)" in result[1]
        pinned_result = client.predict("A", "Text", api_name="/add_pinned")
        assert "1 pinned token" in pinned_result[2]

        out = client.predict(
            "Completion", "Hello world", "[]", "", "Template default", "",
            6, "greedy", 1.0,
            "Jacobian", 0, -1, 1, 8, False,
            api_name="/generate_and_analyze",
        )
        readout_html, _dist, _heatmap, _pinned, status, text = out[:6]
        # Status names each edit and its basis (replaces the old bare count).
        assert "Interventions:" in status
        assert "steer" in status and "@L0" in status and "@L1" in status
        assert "(jacobian)" in status and "(logit)" in status
        # No mismatch warning: @L0 jacobian matches the Jacobian view, and the
        # @L1 logit edit is on the final layer (2-layer model) — basis-exempt.
        assert "⚠" not in status
        # Both steers compose; the final-layer one dominates greedy decoding,
        # so the generated text must consist of steered tokens.
        generated = text[len("Hello world"):]
        assert generated and set(generated) <= {"A", "B"}
        # ...and both steered tokens surface prominently in the summary
        # (server-rendered HTML table cells).
        assert ">A<" in readout_html and ">B<" in readout_html
        # The other views render lazily from the cached slice when their tab
        # is opened; both edited layers carry the ⚡ marker in the heatmap.
        counts = client.predict(api_name="/open_readouts_view")
        heatmap = client.predict(api_name="/open_heatmap_view")
        pinned = client.predict(api_name="/open_pinned_view")
        assert "<table" in counts and "<table" in heatmap and pinned is not None
        assert "⚡L0" in heatmap and "⚡L1" in heatmap
        # Switching to the Logit view surfaces the mismatch warning for the
        # jacobian-basis @L0 edit (the @L1 logit edit is final-layer exempt).
        # State inputs (analysis/positions/active_view) are excluded from the
        # client signature, so this passes only the lens controls.
        out = client.predict(
            "Logit", 0, -1, 1, 8, False, api_name="/update_readouts"
        )
        status = out[-1]
        assert "⚠" in status and "jacobian basis" in status and "@L0" in status

    def test_active_interventions_table_and_clear(self, lens_app):
        client = lens_app
        added = client.predict(
            "ablate", "C", "Text", "", "Text", 0, 1.0, "logit",
            api_name="/add_intervention",
        )
        assert 'data-miru-iv-action="delete"' in added[0]
        assert 'style="width:100%; border:1px solid rgba(127,127,127,0.18) !important;' in added[0]
        repeated = client.predict(
            "ablate", "C", "Text", "", "Text", 0, 1.0, "logit",
            api_name="/add_intervention",
        )
        assert repeated[0].count('data-miru-iv-row="') == 2
        assert "2 enabled intervention group(s)" in repeated[1]
        from miru_tracer.ui.lens_common import get_active_interventions

        assert len(get_active_interventions()) == 2
        cleared = client.predict(api_name="/clear_interventions")
        assert "No active interventions" in cleared[0]
        assert "Cleared" in cleared[1]
        assert get_active_interventions() == []

    def test_logit_mode_without_fitted_lens_still_works(self, lens_app):
        out = lens_app.predict(
            "Completion", "Hello world", "[]", "", "Template default", "",
            3, "greedy", 1.0,
            "Logit", 0, -1, 1, 5, False,
            api_name="/generate_and_analyze",
        )
        status = out[4]
        assert "logit lens" in status
        assert "<table" in out[0]  # summary table (the eagerly rendered view)
        assert "<table" in lens_app.predict(api_name="/open_heatmap_view")
        assert "<table" in lens_app.predict(api_name="/open_readouts_view")


class TestFitFileManagement:
    def test_status_reports_fitted_lens(self, lens_app):
        status = lens_app.predict(api_name="/fit_file_status")
        assert "tiny/test-model" in status
        assert "averaged over 2 prompts" in status

    def test_upload_validates_and_installs(self, lens_app, tiny_model, tmp_path):
        from gradio_client import handle_file

        from miru_tracer.core._jlens import JacobianLens
        from miru_tracer.core.lens import get_lens_store

        # Re-upload the existing fitted lens through the UI path
        source = get_lens_store().lens_path("tiny/test-model")
        result = lens_app.predict(
            handle_file(str(source)), api_name="/install_fit_file"
        )
        assert "Installed" in result

        # A lens with the wrong d_model must be rejected
        import torch

        wrong = JacobianLens(
            jacobians={0: torch.zeros(8, 8)}, n_prompts=1, d_model=8
        )
        wrong_path = tmp_path / "wrong.pt"
        wrong.save(str(wrong_path))
        result = lens_app.predict(
            handle_file(str(wrong_path)), api_name="/install_fit_file"
        )
        assert "different model" in result
