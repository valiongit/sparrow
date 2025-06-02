# llm/inference_backends.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

class InferenceBackend(ABC):
    def __init__(self, model_path: str, model_kwargs: Optional[Dict] = None, tokenizer_kwargs: Optional[Dict] = None):
        self.model_path = model_path
        self.model_kwargs = model_kwargs if model_kwargs else {}
        self.tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs else {}
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate(self, prompt: str, generation_params: Dict) -> str:
        pass

    @property
    @abstractmethod
    def device(self) -> Any: # Changed to Any to be more generic if non-PyTorch backends are added
        pass


class PyTorchBackend(InferenceBackend):
    def __init__(self, model_path: str, device_preference: str = "cuda_if_available", model_kwargs: Optional[Dict] = None, tokenizer_kwargs: Optional[Dict] = None):
        super().__init__(model_path, model_kwargs, tokenizer_kwargs)
        self._device_preference = device_preference
        self._determined_device: Optional[torch.device] = None
        self._configure_quantization_and_precision()

    def _configure_quantization_and_precision(self):
        """Configures quantization and precision based on model_kwargs."""
        quantization_config = None
        if self.model_kwargs.get("load_in_4bit"):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16, # Default, can be overridden
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("4-bit quantization configured for PyTorch backend.")
        elif self.model_kwargs.get("load_in_8bit"):
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("8-bit quantization configured for PyTorch backend.")
        
        if quantization_config:
            self.model_kwargs["quantization_config"] = quantization_config
        
        # Remove our internal keys if they were only for this setup
        self.model_kwargs.pop("load_in_4bit", None)
        self.model_kwargs.pop("load_in_8bit", None)

        # Handle torch_dtype string to actual torch.dtype
        if "torch_dtype" in self.model_kwargs and isinstance(self.model_kwargs["torch_dtype"], str):
            try:
                self.model_kwargs["torch_dtype"] = getattr(torch, self.model_kwargs["torch_dtype"])
                logger.info(f"PyTorch backend: torch_dtype set to {self.model_kwargs['torch_dtype']}")
            except AttributeError:
                logger.warning(f"Invalid torch_dtype string: {self.model_kwargs['torch_dtype']}. It will be ignored.")
                del self.model_kwargs["torch_dtype"]

    @property
    def device(self) -> torch.device:
        if self._determined_device is None:
            if self._device_preference == "cuda_if_available":
                if torch.cuda.is_available():
                    self._determined_device = torch.device("cuda")
                else:
                    self._determined_device = torch.device("cpu")
                    logger.info("CUDA not available. PyTorch backend falling back to CPU.")
            elif self._device_preference == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA explicitly requested for PyTorch backend, but not available.")
                self._determined_device = torch.device("cuda")
            elif self._device_preference == "cpu":
                self._determined_device = torch.device("cpu")
            else: # Default to CPU if preference is unknown
                logger.warning(f"Unknown device preference '{self._device_preference}' for PyTorch backend. Defaulting to CPU.")
                self._determined_device = torch.device("cpu")
            logger.info(f"PyTorch backend determined device: {self._determined_device}")
        return self._determined_device

    def load_model(self):
        logger.info(f"PyTorch backend: Loading model '{self.model_path}' to device '{self.device}'.")
        logger.debug(f"Model kwargs: {self.model_kwargs}")
        logger.debug(f"Tokenizer kwargs: {self.tokenizer_kwargs}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **self.tokenizer_kwargs)
            
            # For quantized models, device_map="auto" is preferred.
            # Otherwise, load to the determined device.
            device_map_strategy = "auto" if "quantization_config" in self.model_kwargs else self.device

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=device_map_strategy,
                **self.model_kwargs
            )
            
            # If not using device_map="auto" (i.e. no quantization or specific single device target)
            # and model is not yet on the target device (e.g. loaded on meta device first).
            if device_map_strategy != "auto" and hasattr(self.model, 'to') and self.model.device != self.device :
                 self.model.to(self.device)

            self.model.eval()
            logger.info(f"PyTorch backend: Model '{self.model_path}' loaded successfully on {self.model.device if hasattr(self.model, 'device') else 'device_map strategy'}.")
        except Exception as e:
            logger.error(f"PyTorch backend: Failed to load model '{self.model_path}': {e}")
            raise

    def generate(self, prompt: str, generation_params: Dict) -> str:
        if not self.model or not self.tokenizer:
            raise RuntimeError("PyTorch backend: Model and tokenizer must be loaded before generation.")

        logger.debug(f"PyTorch backend generating response for prompt: '{prompt[:100]}...' with params: {generation_params}")
        
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device if hasattr(self.model, 'device') else self.device) for k, v in inputs.items()} # Ensure inputs are on model's device

        final_gen_params = {**generation_params} # Copy to modify
        if 'temp' in final_gen_params and 'temperature' not in final_gen_params:
            final_gen_params['temperature'] = final_gen_params.pop('temp')
        else:
            final_gen_params.pop('temp', None)
        if 'max_tokens' in final_gen_params:
            final_gen_params['max_new_tokens'] = final_gen_params.pop('max_tokens')

        try:
            with torch.no_grad():
                output_sequences = self.model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], **final_gen_params)
            response_ids = output_sequences[0, inputs["input_ids"].shape[-1]:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            logger.debug(f"PyTorch backend generated response: '{response_text[:100]}...'")
            return response_text.strip()
        except Exception as e:
            logger.error(f"PyTorch backend error during generation: {e}")
            raise 