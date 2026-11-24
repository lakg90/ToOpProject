"""Search algorithms for ToOp's DC optimiser, one file per algorithm.

Pick one with the setting ga_config.algorithm. The default "map_elites" is ToOp's original.

To add an algorithm: write a file in this folder with a build(ctx) function,
then add one line to ALGORITHMS below.
"""

from qdax.core.emitters.emitter import Emitter
from toop_engine_topology_optimizer.dc.algorithms import map_elites
from toop_engine_topology_optimizer.dc.algorithms.context import Context

ALGORITHMS = {
    "map_elites": map_elites.build,
}
"""The list of available algorithms: name -> function that builds it."""


def build_algorithm(name: str, ctx: Context) -> Emitter:
    """Look the algorithm up by name and build it.

    Parameters
    ----------
    name : str
        The value of ga_config.algorithm
    ctx : Context
        What the algorithm needs to know about the problem

    Returns
    -------
    Emitter
        The object ToOp's search loop calls to get the next batch of plans

    Raises
    ------
    ValueError
        If no algorithm with that name exists
    """
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {name!r}. Available: {sorted(ALGORITHMS)}")
    return ALGORITHMS[name](ctx)
