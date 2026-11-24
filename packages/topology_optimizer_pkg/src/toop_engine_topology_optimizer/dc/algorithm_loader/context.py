"""Problem information passed to the search algorithms."""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(eq=False)
class Context:
    """What an algorithm gets to know about the problem, built in initialize_genetic_algorithm."""

    batch_size: int
    """Number of topologies per iteration"""

    max_num_splits: int
    """Maximum number of split substations per topology"""

    max_num_disconnections: int
    """Maximum number of disconnected branches per topology"""

    action_set: Any
    """The available busbar splits per substation"""

    n_relevant_subs: int
    """Number of substations that can be split"""

    n_disconnectable: int
    """Number of branches that can be disconnected"""

    mutate_fn: Callable
    """Mutation, (genotypes, key) -> (genotypes, key)"""

    crossover_fn: Callable
    """Crossover, (genotypes_a, genotypes_b, key) -> (genotypes, key)"""

    crossover_mutation_ratio: float
    """The ratio of crossovers to mutations"""

    params: dict = field(default_factory=dict)
    """ga_config.algorithm_params"""
