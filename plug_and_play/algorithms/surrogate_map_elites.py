"""Surrogate-filtered MAP-Elites: STARTER FILE, the model is not written yet.

Sources: Gaier et al. (2018), SAIL; Zhang et al. (2022), Deep Surrogate Assisted MAP-Elites.
Each iteration makes oversample * batch_size mutations, scores them with a cheap model (e.g. the chance of
passing the AC check) and keeps the best batch_size for the DC load flow.

The AC check runs after the search, so the model is trained beforehand, e.g. on the plans in candidates.csv
from earlier runs, and loaded here. As it stands predict returns zeros, so it behaves like plain mutation.

algorithm_params: oversample (default 4), model_file (default None)
"""

from functools import partial

import jax
import jax.numpy as jnp
import structlog
from jaxtyping import Array, Float, PRNGKeyArray
from qdax.core.emitters.emitter import Emitter
from toop_engine_topology_optimizer.dc.algorithm_loader import Context
from toop_engine_topology_optimizer.dc.ga_helpers import TrackingMixingEmitter
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import DiscreteMapElitesRepertoire

logger = structlog.get_logger(__name__)


class SurrogateEmitter(TrackingMixingEmitter):
    """Mutates repertoire members and keeps the mutations the model rates highest."""

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx.mutate_fn, ctx.crossover_fn, 0.0, ctx.batch_size)
        self._ctx = ctx
        self._oversample = int(ctx.params.get("oversample", 4))
        self._model = None  # TODO: load the trained model from ctx.params.get("model_file")

    def predict(self, genotypes: Genotype) -> Float[Array, " n"]:
        """TODO: score each topology with the model, higher is better."""
        return jnp.zeros(genotypes.action_index.shape[0])

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: DiscreteMapElitesRepertoire,
        emitter_state: object,  # noqa: ARG002
        random_key: PRNGKeyArray,
    ) -> tuple[Genotype, dict, PRNGKeyArray]:
        """Make oversample * batch_size mutations and return the batch_size the model rates highest."""
        n = self._ctx.batch_size
        parents, random_key = repertoire.sample(random_key, n * self._oversample)
        children, random_key = self._ctx.mutate_fn(parents, random_key)

        keep = jnp.argsort(-self.predict(children))[:n]
        genotypes = jax.tree_util.tree_map(lambda x: x[keep], children)
        return genotypes, {}, random_key


def build(ctx: Context) -> Emitter:
    """Build the emitter."""
    logger.warning("surrogate_map_elites is a starter file: the model is not written yet")
    return SurrogateEmitter(ctx)
