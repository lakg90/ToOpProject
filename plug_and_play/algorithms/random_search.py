"""Random search: a new batch of random topologies every iteration. Used as a baseline."""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from qdax.core.emitters.emitter import Emitter
from toop_engine_topology_optimizer.dc.algorithm_loader import Context
from toop_engine_topology_optimizer.dc.ga_helpers import TrackingMixingEmitter
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype
from toop_engine_topology_optimizer.dc.genetic_functions.mutation.mutate import create_random_topology
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import DiscreteMapElitesRepertoire


class RandomSearchEmitter(TrackingMixingEmitter):
    """Emits random topologies. init and state_update come from TrackingMixingEmitter, so the counters still work."""

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx.mutate_fn, ctx.crossover_fn, 0.0, ctx.batch_size)
        self._ctx = ctx

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: DiscreteMapElitesRepertoire,
        emitter_state: object,  # noqa: ARG002
        random_key: PRNGKeyArray,
    ) -> tuple[Genotype, dict, PRNGKeyArray]:
        """Emit batch_size random topologies, created as in mutate."""
        ctx = self._ctx

        # Only used for the shapes and the PST taps
        template, random_key = repertoire.sample(random_key, ctx.batch_size)

        random_key, subkey = jax.random.split(random_key)
        _, action, disconnections, _ = jax.vmap(
            lambda key, sub_ids, disc: create_random_topology(
                random_key=key,
                sub_ids=sub_ids,
                disconnections=disc,
                action_set=ctx.action_set,
                n_rel_subs=ctx.n_relevant_subs,
                n_disconnectable_branches=ctx.n_disconnectable,
            )
        )(
            jax.random.split(subkey, ctx.batch_size),
            template.action_index.astype(int),
            template.disconnections.astype(int),
        )

        # Sorted as in mutate, for the uniqueness check
        genotypes = Genotype(
            action_index=jnp.sort(action, axis=1),
            disconnections=jnp.sort(disconnections, axis=1),
            nodal_injections_optimized=template.nodal_injections_optimized,
        )
        return genotypes, {}, random_key


def build(ctx: Context) -> Emitter:
    """Build the random search emitter."""
    return RandomSearchEmitter(ctx)
