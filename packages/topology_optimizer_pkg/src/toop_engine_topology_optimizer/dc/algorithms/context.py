"""What ToOp hands to a search algorithm when it builds it."""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(eq=False)
class Context:
    """Everything an algorithm needs to know about the problem.

    Built once in initialize_genetic_algorithm and passed to the chosen algorithm.
    """

    batch_size: int
    """How many switching plans to suggest per search step."""

    max_num_splits: int
    """The most substations one plan may split."""

    max_num_disconnections: int
    """The most lines one plan may switch off."""

    action_set: Any
    """ToOp's ActionSet: every allowed busbar split, grouped by substation."""

    n_relevant_subs: int
    """Number of substations that can be split."""

    n_disconnectable: int
    """Number of lines that can be switched off."""

    mutate_fn: Callable
    """ToOp's mutation: (plans, key) -> (plans, key). Always returns valid plans."""

    crossover_fn: Callable
    """ToOp's crossover: (plans_a, plans_b, key) -> (plans, key)."""

    crossover_mutation_ratio: float
    """ToOp's setting for the share of crossover against mutation."""

    params: dict = field(default_factory=dict)
    """The algorithm_params setting from ga_config, passed through unchanged."""
