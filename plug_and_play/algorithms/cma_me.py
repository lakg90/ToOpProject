"""Discrete CMA-ME: STARTER FILE, the method itself is not written yet.

Source: Fontaine et al. (2020), Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space.
CMA-ME works on continuous vectors. Our topologies are discrete choices, so the distribution has to be a
discrete one, for example a probability for each action in the action set.

As it stands it mutates random repertoire members (like MAP-Elites without crossover) and its memory only
counts updates and tracks the best fitness, to show how memory works.
"""

from functools import partial

import jax
import jax.numpy as jnp
import structlog
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from qdax.core.emitters.emitter import Emitter
from toop_engine_topology_optimizer.dc.algorithm_loader import Context, EmitterWithMemory, MemoryState
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import DiscreteMapElitesRepertoire

logger = structlog.get_logger(__name__)


class CmaMeEmitter(EmitterWithMemory):
    """Samples topologies from a learned distribution (to be written)."""

    def init_memory(self, random_key: PRNGKeyArray) -> PyTree:  # noqa: ARG002
        """TODO: the starting distribution, e.g. equal probabilities for every action."""
        return {"n_updates": jnp.array(0, dtype=int), "best_fitness": jnp.array(-jnp.inf)}

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: DiscreteMapElitesRepertoire,
        emitter_state: MemoryState,  # noqa: ARG002
        random_key: PRNGKeyArray,
    ) -> tuple[Genotype, dict, PRNGKeyArray]:
        """TODO: sample ctx.batch_size topologies from emitter_state.memory."""
        parents, random_key = repertoire.sample(random_key, self._ctx.batch_size)
        genotypes, random_key = self._ctx.mutate_fn(parents, random_key)
        return genotypes, {}, random_key

    def update_memory(
        self,
        memory: PyTree,
        genotypes: Genotype,  # noqa: ARG002
        fitnesses: Float[Array, " n"],
        repertoire: DiscreteMapElitesRepertoire,  # noqa: ARG002
    ) -> PyTree:
        """TODO: move the distribution towards the topologies that scored best."""
        batch_best = jnp.max(jnp.where(jnp.isfinite(fitnesses), fitnesses, -jnp.inf))
        return {
            "n_updates": memory["n_updates"] + 1,
            "best_fitness": jnp.maximum(memory["best_fitness"], batch_best),
        }


def build(ctx: Context) -> Emitter:
    """Build the emitter."""
    logger.warning("cma_me is a starter file: the method itself is not written yet")
    return CmaMeEmitter(ctx)
