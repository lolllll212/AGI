"""Measurement: the benchmark suite, baselines, drift detection, A/B-tested optimisation."""
from .evaluator import (Change, Comparison, DriftAlarm, DriftMonitor, Evaluation, Evaluator, TaskScore, Tolerance,
                        fingerprint, mcnemar, score_run)
from .extended import EXTENDED
from .optimizer import (Mutation, Optimization, Optimizer, Trial, add_instruction, load_config, propose_instructions,
                        save_config, set_field, with_field)
from .suite import SUITE, EvalReport, EvalResult, EvalTask, has_number, reference_llm, run_suite, write_fixtures

FULL_SUITE = SUITE + EXTENDED            # the eight core tasks plus the sixteen harder ones

__all__ = ["FULL_SUITE", "add_instruction", "Change", "Comparison", "DriftAlarm", "DriftMonitor", "EvalReport",
           "EvalResult", "EvalTask", "Evaluation", "Evaluator", "EXTENDED", "fingerprint",
           "has_number", "load_config", "mcnemar", "Mutation", "Optimization", "Optimizer",
           "propose_instructions", "reference_llm", "run_suite", "save_config", "score_run", "set_field",
           "SUITE", "TaskScore", "Tolerance", "Trial", "with_field", "write_fixtures"]
