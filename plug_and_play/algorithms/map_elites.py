"""MAP-Elites, ToOp's original emitter (mutation and crossover of repertoire members)."""

from qdax.core.emitters.emitter import Emitter
from toop_engine_topology_optimizer.dc.algorithm_loader import Context
from toop_engine_topology_optimizer.dc.ga_helpers import TrackingMixingEmitter


def build(ctx: Context) -> Emitter:
    """Build the TrackingMixingEmitter."""
    return TrackingMixingEmitter(
        ctx.mutate_fn,
        ctx.crossover_fn,
        ctx.crossover_mutation_ratio,
        ctx.batch_size,
    )
