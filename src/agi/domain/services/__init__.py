"""Pure functions over domain data: text parsing, arithmetic, embedding, the static code gate."""
from .arithmetic import evaluate
from .embedding import HashingEmbedder, cosine, keywords, tokens
from .source_check import check_source
from .text import extract_code, extract_json, render_answer

__all__ = ["check_source", "cosine", "evaluate", "extract_code", "extract_json", "HashingEmbedder",
           "keywords", "render_answer", "tokens"]
