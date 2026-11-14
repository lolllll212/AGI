"""Compatibility package: training code now lives in :mod:`agi.reasoning.policy`
(the trained search policy) and :mod:`agi.evaluation.dataset` (fine-tuning data export)."""
from ..evaluation.dataset import export_jsonl, to_chat_example
from ..reasoning.policy import (Comparison, Sample, SearchPolicy, SynthTask, TrainingReport, compare,
                                featurize, make_task, make_tasks, samples_from, train_policy)

__all__ = ["Comparison", "Sample", "SearchPolicy", "SynthTask", "TrainingReport", "compare",
           "export_jsonl", "featurize", "make_task", "make_tasks", "samples_from",
           "to_chat_example", "train_policy"]
