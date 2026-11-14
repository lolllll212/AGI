"""Language-model clients."""
from .openai_compat import OpenAICompatLLM
from .scripted import ScriptedLLM

__all__ = ["OpenAICompatLLM", "ScriptedLLM"]
