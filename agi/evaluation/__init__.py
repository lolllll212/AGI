"""Evaluation: the benchmark suite, baselines and drift detection, A/B-tested optimisation, data export."""
from .dataset import export_jsonl, to_chat_example
from .evaluator import (Change, Comparison, DriftAlarm, DriftMonitor, Evaluation, Evaluator, TaskScore, Tolerance,
                        fingerprint, mcnemar, score_run)
from .optimizer import (Mutation, Optimization, Optimizer, Trial, add_instruction, load_config,
                        propose_instructions, save_config, set_field, with_field)
from .extended import EXTENDED
from .suite import SUITE, EvalReport, EvalResult, EvalTask, has_number, reference_llm, run_suite, write_fixtures

FULL_SUITE = SUITE + EXTENDED            # the eight core tasks plus the sixteen harder ones

__all__ = ["EXTENDED", "FULL_SUITE", "SUITE", "Change", "Comparison", "DriftAlarm", "DriftMonitor", "EvalReport", "EvalResult", "EvalTask",
           "Evaluation", "Evaluator", "Mutation", "Optimization", "Optimizer", "TaskScore", "Tolerance", "Trial",
           "add_instruction", "export_jsonl", "fingerprint", "has_number", "load_config", "mcnemar",
           "propose_instructions", "reference_llm", "run_suite", "save_config", "score_run", "set_field",
           "to_chat_example", "with_field", "write_fixtures"]
