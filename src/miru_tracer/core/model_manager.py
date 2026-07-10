"""Model and tokenizer loading for Miru Tracer."""

from __future__ import annotations

import gc
import os
import threading
import time
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    # transformers >= 5: the broadest multimodal auto-class (superset of
    # image-text-to-text; needed for e.g. Gemma 4's unified multimodal models)
    from transformers import AutoModelForMultimodalLM as AutoModelForMultimodal
except ImportError:  # pragma: no cover - older transformers
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForMultimodal
    except ImportError:  # pragma: no cover - transformers 4.x
        from transformers import AutoModelForVision2Seq as AutoModelForMultimodal

from miru_tracer.core.logging_config import get_logger
from miru_tracer.core.model_runtime import MODEL_RUNTIME_LOCK

logger = get_logger(__name__)


class ModelManager:
    """Manages model and tokenizer loading with singleton pattern for standalone mode."""

    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    _model_name = None
    _lock = threading.Lock()
    _is_loading = False
    _generation = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _invalidate_locked(self) -> tuple[Any, Any, str | None]:
        """Atomically publish an unloaded state and return the old ownership."""
        old = (self._model, self._tokenizer, self._model_name)
        self._model = None
        self._tokenizer = None
        self._device = None
        self._model_name = None
        self._generation += 1
        return old

    @staticmethod
    def _clear_sessions() -> int:
        from miru_tracer.core.session_manager import get_session_manager

        return get_session_manager().clear_all_sessions()

    @staticmethod
    def _clear_model_caches() -> None:
        from miru_tracer.core.lens import clear_model_caches

        clear_model_caches()

    @staticmethod
    def _collect_runtime() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(
        self,
        model_name: str,
        quantization: str = "none",
        trust_remote_code: bool = False,
        minimize_ram_usage: bool = False,
    ) -> tuple[Any, Any, str, dict[str, Any]]:
        """
        Load a model and tokenizer.

        Args:
            model_name: HuggingFace model name
            quantization: "none", "4bit", or "8bit"
            trust_remote_code: Whether to trust remote code (security risk)
            minimize_ram_usage: Use aggressive optimizations to minimize RAM usage during loading

        Returns:
            (model, tokenizer, device, info_dict)

        Raises:
            RuntimeError: If a model load/unload operation is already in progress
        """
        with self._lock:
            if self._is_loading:
                raise RuntimeError(
                    "Model operation already in progress. Please wait for it to complete."
                )
            self._is_loading = True
            old_model, old_tokenizer, old_name = self._invalidate_locked()

        new_model = None
        new_tokenizer = None
        try:
            start_time = time.time()
            logger.info(f"Loading model: {model_name} (quantization={quantization})")
            if old_model is not None:
                logger.warning(f"Clearing previous model from memory: {old_name}")

            self._clear_sessions()
            with MODEL_RUNTIME_LOCK:
                self._clear_model_caches()
                del old_model, old_tokenizer
                self._collect_runtime()

                if torch.cuda.is_available():
                    device = "cuda"
                    device_name = torch.cuda.get_device_name(0)
                    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                    logger.info(
                        f"Device detected: {device_name} (VRAM: {vram:.2f} GB)"
                    )
                else:
                    device = "cpu"
                    device_name = "CPU"
                    vram = 0
                    logger.info("Device detected: CPU (no CUDA available)")

                logger.debug(f"Loading tokenizer for {model_name}")
                new_tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=trust_remote_code
                )
                if new_tokenizer.pad_token is None:
                    new_tokenizer.pad_token = new_tokenizer.eos_token
                vocab_size = len(new_tokenizer)

                load_kwargs = {
                    "device_map": "auto" if torch.cuda.is_available() else None,
                    "trust_remote_code": trust_remote_code,
                    "low_cpu_mem_usage": True,
                }
                quantization_note = None
                if quantization != "none" and not torch.cuda.is_available():
                    quantization_note = (
                        f"{quantization} quantization requires CUDA; "
                        "loaded full precision on CPU instead."
                    )
                    logger.warning(quantization_note)
                    quantization = "none"
                if quantization != "none":
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=(quantization == "8bit"),
                        load_in_4bit=(quantization == "4bit"),
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                else:
                    load_kwargs["dtype"] = (
                        torch.float16 if torch.cuda.is_available() else torch.float32
                    )

                if minimize_ram_usage and torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                    load_kwargs["max_memory"] = {
                        0: f"{int(gpu_mem_gb * 0.8)}GiB",
                        "cpu": "2GiB",
                    }
                    load_kwargs["offload_state_dict"] = True
                    offload_dir = "./model_offload"
                    os.makedirs(offload_dir, exist_ok=True)
                    load_kwargs["offload_folder"] = offload_dir

                is_vlm = False
                try:
                    new_model = AutoModelForCausalLM.from_pretrained(
                        model_name, **load_kwargs
                    )
                except (ValueError, KeyError) as e:
                    if "Unrecognized configuration class" not in str(
                        e
                    ) and "does not support" not in str(e):
                        raise
                    new_model = AutoModelForMultimodal.from_pretrained(
                        model_name, **load_kwargs
                    )
                    is_vlm = True

                if not torch.cuda.is_available() and load_kwargs["device_map"] is None:
                    new_model = new_model.to(device)
                new_model.eval()

                with self._lock:
                    self._model = new_model
                    self._tokenizer = new_tokenizer
                    self._device = device
                    self._model_name = model_name

            # Gather info
            num_params = new_model.num_parameters() / 1e9
            load_time = time.time() - start_time

            info = {
                "model_name": model_name,
                "device": device,
                "device_name": device_name,
                "vram_gb": vram,
                "vocab_size": vocab_size,
                "num_parameters_b": num_params,
                "quantization": quantization,
                "quantization_note": quantization_note,
                "pytorch_version": torch.__version__,
                "is_vlm": is_vlm,
                "vlm_warning": (
                    "Vision-Language Model loaded in text-only mode. Image inputs not supported."
                    if is_vlm
                    else None
                ),
            }

            logger.info(
                f"Model loaded successfully: {num_params:.2f}B parameters in {load_time:.2f}s"
                + (" [VLM - text-only mode]" if is_vlm else "")
            )

            return new_model, new_tokenizer, device, info
        except Exception:
            with self._lock:
                if self._model is new_model or self._tokenizer is new_tokenizer:
                    self._invalidate_locked()
            new_model = new_tokenizer = None
            with MODEL_RUNTIME_LOCK:
                self._clear_model_caches()
                self._collect_runtime()
            raise
        finally:
            with self._lock:
                self._is_loading = False

    def get_model(self) -> Any | None:
        """Get currently loaded model."""
        return self._model

    def get_tokenizer(self) -> Any | None:
        """Get currently loaded tokenizer."""
        return self._tokenizer

    def get_device(self) -> str | None:
        """Get current device."""
        return self._device

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None and self._tokenizer is not None

    def get_model_name(self) -> str | None:
        """Get currently loaded model name."""
        return self._model_name

    def get_generation(self) -> int:
        """Monotonic identifier invalidating state from earlier models."""
        return self._generation

    def snapshot(self) -> tuple[Any, Any, str, int] | None:
        """Return one consistent loaded-model snapshot, or None during transition."""
        with self._lock:
            if self._is_loading or self._model is None or self._tokenizer is None:
                return None
            return self._model, self._tokenizer, self._device, self._generation

    def unload_model(self) -> dict[str, Any]:
        """
        Unload the currently loaded model and free memory.

        Returns:
            Dictionary with status information

        Raises:
            RuntimeError: If a model load/unload operation is already in progress
        """
        with self._lock:
            if self._is_loading:
                raise RuntimeError(
                    "Model operation already in progress. Please wait for it to complete."
                )
            if self._model is None and self._tokenizer is None:
                return {"status": "warning", "message": "No model currently loaded"}
            self._is_loading = True
            old_model, old_tokenizer, model_name = self._invalidate_locked()

        try:
            if old_model is not None:
                logger.info(f"Unloading model: {model_name}")
            cleared_sessions = self._clear_sessions()
            with MODEL_RUNTIME_LOCK:
                self._clear_model_caches()
                del old_model, old_tokenizer
                self._collect_runtime()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    logger.debug("GPU cache cleared and synchronized")

            logger.info(f"Model unloaded successfully: {model_name}")

            return {
                "status": "success",
                "message": f"Model '{model_name}' unloaded successfully",
                "previous_model": model_name,
                "cleared_sessions": cleared_sessions,
            }
        finally:
            with self._lock:
                self._is_loading = False

    def is_loading(self) -> bool:
        """Check if a model load/unload operation is in progress."""
        with self._lock:
            return self._is_loading
