"""MAP-Elites: ToOp's own search, unchanged.

Each step it picks plans already kept in the repertoire and changes them a little
(mutation) or mixes two of them (crossover).
"""

from qdax.core.emitters.emitter import Emitter
from toop_engine_topology_optimizer.dc.algorithms.context import Context
from toop_engine_topology_optimizer.dc.ga_helpers import TrackingMixingEmitter


def build(ctx: Context) -> Emitter:
    """Build ToOp's original emitter with exactly the arguments ToOp used before."""
    return TrackingMixingEmitter(
        ctx.mutate_fn,
        ctx.crossover_fn,
        ctx.crossover_mutation_ratio,
        ctx.batch_size,
    )
