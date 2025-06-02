import warnings
import typer
from typing_extensions import Annotated, List
from pipelines.interface import get_pipeline
import tempfile
import os
from rich import print
import logging
import json
from llm.inference_backends import PyTorchBackend, InferenceBackend
from llm.config_utils import AppConfig


# Disable parallelism in the Huggingface tokenizers library to prevent potential deadlocks and ensure consistent behavior.
# This is especially important in environments where multiprocessing is used, as forking after parallelism can lead to issues.
# Note: Disabling parallelism may impact performance, but it ensures safer and more predictable execution.
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)
app_config = AppConfig()


def run(query: Annotated[str, typer.Argument(help="The list of fields to fetch")],
        file_path: Annotated[str, typer.Option(help="The file to process")] = None,
        pipeline: Annotated[str, typer.Option(help="Selected pipeline")] = "sparrow-parse",
        options: Annotated[List[str], typer.Option(help="Options to pass to the pipeline")] = None,
        crop_size: Annotated[int, typer.Option(help="Crop size for table extraction")] = None,
        page_type: Annotated[List[str], typer.Option(help="Page type query")] = None,
        debug_dir: Annotated[str, typer.Option(help="Debug folder for multipage")] = None,
        debug: Annotated[bool, typer.Option(help="Enable debug mode")] = False):

    user_selected_pipeline = pipeline  # Modify this as needed

    try:
        rag = get_pipeline(user_selected_pipeline)
        answer = rag.run_pipeline(user_selected_pipeline, query, file_path, options, crop_size, page_type,
                                  debug_dir, debug, False)

        print(f"\nSparrow response:\n")
        print(answer)
    except ValueError as e:
        print(f"Caught an exception: {e}")


async def run_from_api_engine(user_selected_pipeline, query, options_arr, crop_size, page_type, file, debug_dir, debug):
    try:
        rag = get_pipeline(user_selected_pipeline)

        if file is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, file.filename)

                # Save the uploaded file to the temporary directory
                with open(temp_file_path, 'wb') as temp_file:
                    content = await file.read()
                    temp_file.write(content)

                answer = rag.run_pipeline(user_selected_pipeline, query, temp_file_path, options_arr, crop_size, page_type,
                                          debug_dir, debug, False)
        else:
            answer = rag.run_pipeline(user_selected_pipeline, query, None, options_arr, crop_size, page_type,
                                      debug_dir, debug, False)
    except ValueError as e:
        raise e

    return answer


# Add a new function for instruction-only processing
async def run_from_api_engine_instruction(user_selected_pipeline, query, options_arr, debug_dir, debug):
    """
    Instruction-only version of run_from_api_engine that doesn't require a file.
    """
    try:
        rag = get_pipeline(user_selected_pipeline)

        # Call run_pipeline with file_path=None for instruction-only processing
        answer = rag.run_pipeline(
            user_selected_pipeline,
            query,
            None,  # No file path for instruction-only queries
            options_arr,
            None,  # No crop_size needed
            None,  # No page_type needed
            debug_dir,
            debug,
            False
        )
    except ValueError as e:
        raise e

    return answer


class LLMEngine:
    def __init__(self, model_name: str, options_list: List[str]):
        """
        Initializes the LLM engine with a specified model and options.
        Dynamically selects an inference backend (PyTorch CUDA/CPU) based on configuration.

        Args:
            model_name (str): The query or prompt template. This is NOT the model path.
            options_list (List[str]): A list of options strings.
                The model path is expected to be the first or second element.
                Other elements can be key-value pairs for model/tokenizer configuration,
                e.g., "quantize:4bit", "precision:float16", "trust_remote_code:true".
        """
        self.model_name = model_name
        self.options_list = options_list
        self.model = None
        self.tokenizer = None
        self.model_type = "mlx" # Default or determined by options_list[0]
        self.backend: Optional[InferenceBackend] = None

        if not options_list:
            raise ValueError("options_list cannot be empty as it must contain the model path.")

        actual_model_path = ""
        kv_options_start_index = 0

        # Determine model path from options_list
        # It's typically the first element containing '/' or the second if the first was an old backend hint.
        if "/" in self.options_list[0]:
            actual_model_path = self.options_list[0]
            kv_options_start_index = 1
        elif len(self.options_list) > 1 and "/" in self.options_list[1]:
            # Covers old format like ["mlx", "org/model", "quantize:4bit"]
            actual_model_path = self.options_list[1]
            kv_options_start_index = 2
        elif self.options_list[0].lower() not in ["mlx", "pytorch", "cpu", "cuda"]: # Check if first item is not an old hint
            actual_model_path = self.options_list[0] # Assume it's a model path/alias
            kv_options_start_index = 1
        else:
            raise ValueError(f"Could not determine model path from options_list: {self.options_list}")

        if not actual_model_path:
             raise ValueError(f"Model path is empty. Could not parse from options_list: {self.options_list}")
        
        logger.info(f"Determined model path for LLMEngine: {actual_model_path}")

        parsed_kv_options = {}
        for opt_str in self.options_list[kv_options_start_index:]:
            if ":" in opt_str:
                key, value = opt_str.split(":", 1)
                parsed_kv_options[key.strip().lower()] = value.strip()
            else:
                logger.warning(f"Option '{opt_str}' from options_list is not in key:value format and will be ignored for model/tokenizer kwargs.")

        model_kwargs = {}
        tokenizer_kwargs = {}

        if parsed_kv_options.get("trust_remote_code", "false").lower() == "true":
            model_kwargs["trust_remote_code"] = True
            tokenizer_kwargs["trust_remote_code"] = True

        if parsed_kv_options.get("quantize") == "4bit":
            model_kwargs["load_in_4bit"] = True
        elif parsed_kv_options.get("quantize") == "8bit":
            model_kwargs["load_in_8bit"] = True

        if parsed_kv_options.get("precision") == "float16":
            model_kwargs["torch_dtype"] = "float16"
        elif parsed_kv_options.get("precision") == "bfloat16":
            model_kwargs["torch_dtype"] = "bfloat16"
        
        if parsed_kv_options.get("use_fast_tokenizer", "true").lower() == "false":
            tokenizer_kwargs["use_fast"] = False

        backend_preference = app_config.get_config_value("llm_engine", "inference_backend", "pytorch_cuda_if_available")
        logger.info(f"LLMEngine selected backend preference: {backend_preference}")

        if backend_preference.startswith("pytorch"):
            device_pref = backend_preference.split("_", 1)[1] if "_" in backend_preference else "cuda_if_available"
            self.backend = PyTorchBackend(
                model_path=actual_model_path, 
                device_preference=device_pref,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs
            )
        else:
            raise ValueError(f"Unsupported backend preference: {backend_preference}. Only 'pytorch_*' variants are currently supported.")

        try:
            self.backend.load_model()
            logger.info(f"LLMEngine initialized successfully with model '{actual_model_path}' using backend '{self.backend.__class__.__name__}'.")
        except Exception as e:
            logger.error(f"Failed to initialize LLMEngine with model '{actual_model_path}': {e}")
            self.backend = None 
            raise

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates text based on the provided prompt.

        Args:
            prompt (str): The input prompt for text generation.
            **kwargs: Additional generation parameters.

        Returns:
            str: The generated text.
        """
        if not self.backend:
            raise RuntimeError("LLM backend is not initialized. Cannot generate.")

        logger.debug(f"LLMEngine.generate called with prompt: '{prompt[:100]}...' and kwargs: {kwargs}")
        
        try:
            return self.backend.generate(prompt, generation_params=kwargs)
        except Exception as e:
            logger.error(f"Error during generation in LLMEngine: {e}")
            raise


if __name__ == "__main__":
    typer.run(run)
