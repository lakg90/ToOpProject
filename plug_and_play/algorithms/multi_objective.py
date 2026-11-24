"""Multi-objective search (towards NSGA-II and MOME): STARTER FILE, the selection is not written yet.

Sources: Deb et al. (2002), NSGA-II; Pierrot et al. (2022), Multi-Objective Quality Diversity (MOME).
The repertoire stores every metric for each kept topology (repertoire.extra_scores), so the objectives can
be read from there.

As it stands it picks parents only from the topologies no other kept topology beats on every objective
(the first Pareto front) and mutates them. NSGA-II's ranking of later fronts and crowding distance are
not written yet.

Note: what goes to the AC check is still ToOp's repertoire, ranked by ToOp's combined fitness.

algorithm_params: objectives, the metrics to minimise
    (default overload_energy_artificial_limit_n_1 and switching_distance)
"""

from functools import partial

import jax
import jax.numpy as jnp
import structlog
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from qdax.core.emitters.emitter import Emitter
from toop_engine_topology_optimizer.dc.algorithm_loader import Context
from toop_engine_topology_optimizer.dc.ga_helpers import TrackingMixingEmitter
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import DiscreteMapElitesRepertoire

logger = structlog.get_logger(__name__)

DEFAULT_OBJECTIVES = ("overload_energy_artificial_limit_n_1", "switching_distance")


def first_front(objectives: Float[Array, " n k"], valid: Bool[Array, " n"]) -> Bool[Array, " n"]:
    """Mark the valid rows that no other valid row beats on every objective (lower is better)."""
    no_worse = jnp.all(objectives[:, None, :] <= objectives[None, :, :], axis=-1)  # [j, i]: j no worse than i
    better = jnp.any(objectives[:, None, :] < objectives[None, :, :], axis=-1)
    dominated = jnp.any(no_worse & better & valid[:, None], axis=0)
    return valid & ~dominated


class MultiObjectiveEmitter(TrackingMixingEmitter):
    """Mutates topologies from the first Pareto front of the repertoire."""

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx.mutate_fn, ctx.crossover_fn, 0.0, ctx.batch_size)
        self._ctx = ctx
        self._objectives = tuple(ctx.params.get("objectives", DEFAULT_OBJECTIVES))

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: DiscreteMapElitesRepertoire,
        emitter_state: object,  # noqa: ARG002
        random_key: PRNGKeyArray,
    ) -> tuple[Genotype, dict, PRNGKeyArray]:
        """Pick parents from the first front, then mutate them."""
        objectives = jnp.stack([repertoire.extra_scores[name] for name in self._objectives], axis=-1)
        front = first_front(objectives, jnp.isfinite(repertoire.fitnesses))

        # TODO: NSGA-II would rank all fronts and prefer spread-out topologies (crowding distance)
        random_key, subkey = jax.random.split(random_key)
        parent_index = jax.random.choice(subkey, front.shape[0], shape=(self._ctx.batch_size,), p=front / jnp.sum(front))
        parents = jax.tree_util.tree_map(lambda x: x[parent_index], repertoire.genotypes)
        genotypes, random_key = self._ctx.mutate_fn(parents, random_key)
        return genotypes, {}, random_key


def build(ctx: Context) -> Emitter:
    """Build the emitter."""
    logger.warning("multi_objective is a starter file: NSGA-II and MOME selection are not written yet")
    return MultiObjectiveEmitter(ctx)
