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
    # transformers >= 5 renamed AutoModelForVision2Seq
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - transformers 4.x
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

from miru_tracer.core.logging_config import get_logger

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

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

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

        try:
            start_time = time.time()
            logger.info(f"Loading model: {model_name} (quantization={quantization})")

            # Clear previous model if exists
            if self._model is not None:
                logger.warning(f"Clearing previous model from memory: {self._model_name}")
                del self._model
                self._model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
                device_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"Device detected: {device_name} (VRAM: {vram:.2f} GB)")
            else:
                self._device = "cpu"
                device_name = "CPU"
                vram = 0
                logger.info("Device detected: CPU (no CUDA available)")

            # Load tokenizer
            logger.debug(f"Loading tokenizer for {model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )

            # Handle padding token
            if self._tokenizer.pad_token is None:
                logger.debug("Setting pad_token to eos_token (pad_token was None)")
                self._tokenizer.pad_token = self._tokenizer.eos_token

            vocab_size = len(self._tokenizer)
            logger.info(f"Tokenizer loaded: vocab_size={vocab_size:,}")

            # Build common loading kwargs
            load_kwargs = {
                "device_map": "auto" if torch.cuda.is_available() else None,
                "trust_remote_code": trust_remote_code,
                "low_cpu_mem_usage": True,  # Critical: load directly to GPU, minimal RAM
            }

            # Quantization needs CUDA (bitsandbytes); tell the user instead of
            # silently loading full precision.
            quantization_note = None
            if quantization != "none" and not torch.cuda.is_available():
                quantization_note = (
                    f"{quantization} quantization requires CUDA; "
                    "loaded full precision on CPU instead."
                )
                logger.warning(quantization_note)
                quantization = "none"

            # Add quantization or dtype
            if quantization != "none":
                logger.debug(
                    f"Using quantization: {quantization} (load_in_{quantization[0]}bit=True)"
                )
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=(quantization == "8bit"),
                    load_in_4bit=(quantization == "4bit"),
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["quantization_config"] = quantization_config
            else:
                # Set dtype based on available hardware
                model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                load_kwargs["dtype"] = model_dtype
                logger.debug(
                    f"Loading model with dtype={model_dtype}, device_map={load_kwargs['device_map']}"
                )

            # Apply RAM optimization if requested
            if minimize_ram_usage and torch.cuda.is_available():
                logger.info(
                    "Applying RAM optimization settings (slower loading, minimal RAM usage)"
                )
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                # Reserve 80% of GPU memory, limit CPU RAM to 2GB
                load_kwargs["max_memory"] = {0: f"{int(gpu_mem_gb * 0.8)}GiB", "cpu": "2GiB"}
                # Stream weights directly to device without full RAM allocation
                load_kwargs["offload_state_dict"] = True
                offload_dir = "./model_offload"
                os.makedirs(offload_dir, exist_ok=True)
                load_kwargs["offload_folder"] = offload_dir
                logger.debug(
                    f"RAM optimization: max_memory={load_kwargs['max_memory']}, "
                    f"offload_folder={offload_dir}"
                )

            # Try loading as CausalLM, fallback to image-text-to-text for VLMs
            is_vlm = False
            try:
                logger.debug("Attempting to load with AutoModelForCausalLM")
                self._model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            except (ValueError, KeyError) as e:
                if "Unrecognized configuration class" in str(e) or "does not support" in str(e):
                    logger.warning(
                        "AutoModelForCausalLM failed. Retrying as image-text-to-text "
                        "model (VLM detected)..."
                    )
                    self._model = AutoModelForImageTextToText.from_pretrained(
                        model_name, **load_kwargs
                    )
                    is_vlm = True
                    logger.info(
                        "Loaded Vision-Language Model in text-only mode. "
                        "Image inputs not supported."
                    )
                else:
                    raise

            # If CPU mode and not using device_map, explicitly move to device
            if not torch.cuda.is_available() and load_kwargs["device_map"] is None:
                logger.debug("Moving model to CPU device")
                self._model = self._model.to(self._device)

            self._model.eval()
            self._model_name = model_name

            # Gather info
            num_params = self._model.num_parameters() / 1e9
            load_time = time.time() - start_time

            info = {
                "model_name": model_name,
                "device": self._device,
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

            return self._model, self._tokenizer, self._device, info
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
            self._is_loading = True

        try:
            if self._model is None and self._tokenizer is None:
                return {"status": "warning", "message": "No model currently loaded"}

            model_name = self._model_name

            if self._model is not None:
                logger.info(f"Unloading model: {model_name}")
                del self._model
                self._model = None

            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None

            self._device = None
            self._model_name = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for GPU operations to complete
                logger.debug("GPU cache cleared and synchronized")

            logger.info(f"Model unloaded successfully: {model_name}")

            return {
                "status": "success",
                "message": f"Model '{model_name}' unloaded successfully",
                "previous_model": model_name,
            }
        finally:
            with self._lock:
                self._is_loading = False

    def is_loading(self) -> bool:
        """Check if a model load/unload operation is in progress."""
        with self._lock:
            return self._is_loading
