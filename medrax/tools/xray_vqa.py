from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
from pydantic import BaseModel, Field

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


class XRayVQAToolInput(BaseModel):
    """Input schema for the CheXagent Tool."""

    image_paths: List[str] = Field(
        ..., description="List of paths to chest X-ray images to analyze"
    )
    prompt: str = Field(..., description="Question or instruction about the chest X-ray images")
    max_new_tokens: int = Field(
        512, description="Maximum number of tokens to generate in the response"
    )


class XRayVQATool(BaseTool):
    """Tool that leverages CheXagent for comprehensive chest X-ray analysis."""

    name: str = "chest_xray_expert"
    description: str = (
        "A versatile tool for analyzing chest X-rays. "
        "Can perform multiple tasks including: visual question answering, report generation, "
        "abnormality detection, comparative analysis, anatomical description, "
        "and clinical interpretation. Input should be paths to X-ray images "
        "and a natural language prompt describing the analysis needed."
    )
    args_schema: Type[BaseModel] = XRayVQAToolInput
    return_direct: bool = True
    cache_dir: Optional[str] = None
    device: Optional[str] = None
    dtype: torch.dtype = torch.bfloat16
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None

    def __init__(
        self,
        model_name: str = "StanfordAIMI/CheXagent-2-3b",
        device: Optional[str] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the XRayVQATool.

        Args:
            model_name: Name of the CheXagent model to use
            device: Device to run model on (cuda/cpu)
            dtype: Data type for model weights
            cache_dir: Directory to cache downloaded models
            load_in_4bit: Use 4-bit quantization (reduces VRAM from ~6GB to ~2GB)
            load_in_8bit: Use 8-bit quantization
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)

        import transformers
        from transformers import BitsAndBytesConfig

        original_transformers_version = transformers.__version__
        transformers.__version__ = "4.40.0"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.cache_dir = cache_dir

        # Download model and patch version assert in remote code for compatibility
        local_path = self._download_and_patch_model(model_name, cache_dir)

        # Setup quantization config
        quant_kwargs = {}
        if load_in_4bit:
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load tokenizer and model from (patched) local path
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path,
            device_map="auto" if quant_kwargs else str(self.device),
            trust_remote_code=True,
            cache_dir=cache_dir,
            **quant_kwargs,
        )
        if not (load_in_4bit or load_in_8bit):
            self.model = self.model.to(dtype=self.dtype)
        self.model.eval()

        transformers.__version__ = original_transformers_version

    @staticmethod
    def _download_and_patch_model(model_name: str, cache_dir: Optional[str] = None) -> str:
        """Download model and patch version-check asserts for transformers compatibility.

        Accepts either a HuggingFace repo ID (downloaded via snapshot_download) or a
        local directory path (e.g. a Kaggle dataset mount), which is used directly.
        """
        import glob
        import os

        # If model_name is already a local directory (e.g. Kaggle dataset mount),
        # skip snapshot_download — it only accepts HF repo IDs and would crash.
        if os.path.isdir(model_name):
            local_path = model_name
        else:
            from huggingface_hub import snapshot_download
            local_path = snapshot_download(model_name, cache_dir=cache_dir)

        # Patch any version assert in the remote modeling code
        for py_file in glob.glob(str(Path(local_path) / "*.py")):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                if 'assert transformers.__version__' in content:
                    content = content.replace(
                        'assert transformers.__version__ == "4.40.0"',
                        '# assert transformers.__version__ == "4.40.0"  # Patched for compat',
                    )
                    with open(py_file, "w", encoding="utf-8") as f:
                        f.write(content)
            except Exception:
                pass  # Read-only or inaccessible file, skip

        return local_path

    def _generate_response(self, image_paths: List[str], prompt: str, max_new_tokens: int,
                            do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0) -> str:
        """Generate response using CheXagent model.

        Args:
            image_paths: List of paths to chest X-ray images
            prompt: Question or instruction about the images
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        Returns:
            str: Model's response
        """
        query = self.tokenizer.from_list_format(
            [*[{"image": path} for path in image_paths], {"text": prompt}]
        )
        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        ).to(device=self.device)

        # Run inference
        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                do_sample=do_sample,
                num_beams=1,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                max_new_tokens=max_new_tokens,
            )[0]
            response = self.tokenizer.decode(output[input_ids.size(1) : -1])

            return response

    def _run(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Execute the chest X-ray analysis.

        Args:
            image_paths: List of paths to chest X-ray images
            prompt: Question or instruction about the images
            max_new_tokens: Maximum number of tokens to generate
            run_manager: Optional callback manager

        Returns:
            Tuple[Dict[str, Any], Dict]: Output dictionary and metadata dictionary
        """
        try:
            # Verify image paths
            for path in image_paths:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Image file not found: {path}")

            response = self._generate_response(image_paths, prompt, max_new_tokens)

            output = {
                "response": response,
            }

            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "completed",
            }

            return output, metadata

        except Exception as e:
            output = {"error": str(e)}
            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "failed",
                "error_details": str(e),
            }
            return output, metadata

    async def _arun(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 256,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_paths, prompt, max_new_tokens)
