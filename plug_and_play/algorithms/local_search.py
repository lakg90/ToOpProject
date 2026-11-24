"""Local search: mutate the best topology so far.

A share restart_prob (default 0.2) of each batch starts from a random repertoire member instead.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from qdax.core.emitters.emitter import Emitter
from toop_engine_topology_optimizer.dc.algorithm_loader import Context
from toop_engine_topology_optimizer.dc.ga_helpers import TrackingMixingEmitter
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import DiscreteMapElitesRepertoire


class LocalSearchEmitter(TrackingMixingEmitter):
    """Emits mutations of the best topology in the repertoire."""

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx.mutate_fn, ctx.crossover_fn, 0.0, ctx.batch_size)
        self._ctx = ctx
        self._restart_prob = float(ctx.params.get("restart_prob", 0.2))

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: DiscreteMapElitesRepertoire,
        emitter_state: object,  # noqa: ARG002
        random_key: PRNGKeyArray,
    ) -> tuple[Genotype, dict, PRNGKeyArray]:
        """Emit batch_size mutations of the best topology, some of them of random members instead."""
        n = self._ctx.batch_size

        best_index = jnp.argmax(repertoire.fitnesses)
        best = jax.tree_util.tree_map(lambda x: jnp.repeat(x[best_index][None], n, axis=0), repertoire.genotypes)
        others, random_key = repertoire.sample(random_key, n)

        random_key, subkey = jax.random.split(random_key)
        restart = jax.random.uniform(subkey, (n,)) < self._restart_prob
        parents = jax.tree_util.tree_map(
            lambda b, o: jnp.where(restart.reshape((n,) + (1,) * (b.ndim - 1)), o, b),
            best,
            others,
        )

        genotypes, random_key = self._ctx.mutate_fn(parents, random_key)
        return genotypes, {}, random_key


def build(ctx: Context) -> Emitter:
    """Build the local search emitter."""
    return LocalSearchEmitter(ctx)
