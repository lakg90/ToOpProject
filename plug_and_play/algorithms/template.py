"""Template for a new algorithm: copy, rename and fill in emit. Runs as is (mutates random members)."""

from functools import partial

import jax
from jaxtyping import PRNGKeyArray
from qdax.core.emitters.emitter import Emitter
from toop_engine_topology_optimizer.dc.algorithm_loader import Context
from toop_engine_topology_optimizer.dc.ga_helpers import TrackingMixingEmitter
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import DiscreteMapElitesRepertoire


class MyEmitter(TrackingMixingEmitter):
    """Describe your algorithm here."""

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx.mutate_fn, ctx.crossover_fn, 0.0, ctx.batch_size)
        self._ctx = ctx
        # Read your own settings here, e.g. self._step = ctx.params.get("step", 1)

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: DiscreteMapElitesRepertoire,
        emitter_state: object,  # noqa: ARG002
        random_key: PRNGKeyArray,
    ) -> tuple[Genotype, dict, PRNGKeyArray]:
        """Return exactly ctx.batch_size new topologies and a new random key."""
        n = self._ctx.batch_size

        # Replace these two lines with your algorithm
        parents, random_key = repertoire.sample(random_key, n)
        genotypes, random_key = self._ctx.mutate_fn(parents, random_key)

        return genotypes, {}, random_key


def build(ctx: Context) -> Emitter:
    """Build the emitter. Add this function to ALGORITHMS in __init__.py."""
    return MyEmitter(ctx)
