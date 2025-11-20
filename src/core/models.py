"""Data models and model loading utilities for Miru Tracer."""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
import time
from core.logging_config import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


@dataclass
class TokenStep:
    """Records information about a single token generation step."""

    step: int
    token_id: int
    token_text: str
    probability: float
    top_k_tokens: List[int]
    top_k_probs: List[float]
    top_k_texts: List[str]
    all_logits: Optional[List[float]] = None  # Full vocabulary logits (optional)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ModelManager:
    """Manages model and tokenizer loading with singleton pattern for standalone mode."""

    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    _model_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def load_model(
        self,
        model_name: str,
        quantization: str = "none",
        trust_remote_code: bool = False,
    ) -> Tuple[Any, Any, str, Dict[str, Any]]:
        """
        Load a model and tokenizer.

        Args:
            model_name: HuggingFace model name
            quantization: "none", "4bit", or "8bit"
            trust_remote_code: Whether to trust remote code (security risk)

        Returns:
            (model, tokenizer, device, info_dict)
        """
        start_time = time.time()
        logger.info(f"Loading model: {model_name} (quantization={quantization})")

        # Clear previous model if exists
        if self._model is not None:
            logger.warning(f"Clearing previous model from memory: {self._model_name}")
            del self._model
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

        # Load model with optional quantization
        if quantization != "none" and torch.cuda.is_available():
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
            logger.debug(f"Loading model with device_map='auto'")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=trust_remote_code,
            )
        else:
            # Determine dtype and device map based on available hardware
            if torch.cuda.is_available():
                model_dtype = torch.float16
                model_device_map = "auto"
                logger.debug(
                    f"Loading model with dtype={model_dtype}, device_map='auto'"
                )
            else:
                model_dtype = torch.float32
                model_device_map = None
                logger.debug(
                    f"Loading model with dtype={model_dtype}, device_map=None (CPU mode)"
                )

            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=model_dtype,
                device_map=model_device_map,
                trust_remote_code=trust_remote_code,
            )

            # If CPU, explicitly move to device
            if not torch.cuda.is_available():
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
            "vocab_size": len(self._tokenizer),
            "num_parameters_b": num_params,
            "quantization": quantization,
            "pytorch_version": torch.__version__,
        }

        logger.info(
            f"Model loaded successfully: {num_params:.2f}B parameters in {load_time:.2f}s"
        )

        return self._model, self._tokenizer, self._device, info

    def get_model(self) -> Optional[Any]:
        """Get currently loaded model."""
        return self._model

    def get_tokenizer(self) -> Optional[Any]:
        """Get currently loaded tokenizer."""
        return self._tokenizer

    def get_device(self) -> Optional[str]:
        """Get current device."""
        return self._device

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None and self._tokenizer is not None
