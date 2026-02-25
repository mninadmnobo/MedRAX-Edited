"""Lazy-loading tool wrappers for memory-constrained GPU environments.

Provides ``LazyTool`` – a LangChain-compatible wrapper that defers model
loading until the tool is actually invoked – and ``GPUMemoryManager`` which
ensures only one heavyweight tool is resident on a target GPU at any time.

Typical use on 2×T4 Kaggle setup:

    gpu1_mgr = GPUMemoryManager()

    lazy_vqa = LazyTool(XRayVQATool, gpu1_mgr, model_name="...", device="cuda:1")
    lazy_llava = LazyTool(LlavaMedTool, gpu1_mgr, model_path="...", device="cuda:1")

    agent = Agent(model, tools=[classifier, lazy_vqa, lazy_llava], ...)

When the agent calls ``lazy_vqa``, GPUMemoryManager unloads whatever was on
GPU 1 and instantiates XRayVQATool.  When it later calls ``lazy_llava``, the
VQA model is released first.
"""

import gc
from typing import Any, Dict, Optional, Type

import torch
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, PrivateAttr


class GPUMemoryManager:
    """Tracks which LazyTool is currently loaded and unloads it before activating another."""

    def __init__(self):
        self._active: Optional["LazyTool"] = None

    def activate(self, tool: "LazyTool") -> None:
        """Ensure *tool* is loaded, unloading the current occupant if different."""
        if self._active is tool and tool._real_tool is not None:
            return
        if self._active is not None and self._active is not tool:
            print(f"[GPU-Mgr] Unloading {self._active.name}")
            self._active._unload()
        print(f"[GPU-Mgr] Loading   {tool.name}")
        tool._load()
        self._active = tool

    def unload_all(self) -> None:
        if self._active is not None:
            self._active._unload()
            self._active = None


def _field_default(cls: type, name: str, fallback: Any = None) -> Any:
    """Get a Pydantic field default from either v2 or v1 style."""
    # Pydantic v2
    if hasattr(cls, "model_fields"):
        f = cls.model_fields.get(name)
        return f.default if f is not None else fallback
    # Pydantic v1
    if hasattr(cls, "__fields__"):
        f = cls.__fields__.get(name)
        return f.default if f is not None else fallback
    return fallback


class LazyTool(BaseTool):
    """LangChain tool wrapper that defers heavyweight model loading to first call.

    The wrapper copies *name*, *description*, and *args_schema* from the real
    tool class so the LLM agent can see the tool's signature without loading
    any model weights.
    """

    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] = BaseModel
    return_direct: bool = False

    _tool_cls: Any = PrivateAttr()
    _tool_kwargs: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _real_tool: Optional[BaseTool] = PrivateAttr(default=None)
    _manager: GPUMemoryManager = PrivateAttr()

    def __init__(self, tool_cls: type, manager: GPUMemoryManager, **tool_kwargs: Any):
        super().__init__(
            name=_field_default(tool_cls, "name", "unknown_tool"),
            description=_field_default(tool_cls, "description", ""),
            args_schema=_field_default(tool_cls, "args_schema", BaseModel),
            return_direct=_field_default(tool_cls, "return_direct", False),
        )
        self._tool_cls = tool_cls
        self._tool_kwargs = tool_kwargs
        self._manager = manager

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._real_tool is None:
            self._real_tool = self._tool_cls(**self._tool_kwargs)

    def _unload(self) -> None:
        if self._real_tool is None:
            return
        # Release large tensor attributes
        for attr in (
            "model", "tokenizer", "processor", "image_processor",
            "findings_model", "impression_model",
            "findings_tokenizer", "impression_tokenizer",
            "findings_processor", "impression_processor",
        ):
            if hasattr(self._real_tool, attr):
                try:
                    setattr(self._real_tool, attr, None)
                except Exception:
                    pass
        self._real_tool = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        self._manager.activate(self)
        return self._real_tool._run(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        self._manager.activate(self)
        return await self._real_tool._arun(*args, **kwargs)
