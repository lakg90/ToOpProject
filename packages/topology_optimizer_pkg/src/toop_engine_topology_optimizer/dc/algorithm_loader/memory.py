"""Base class for algorithms that keep their own memory between iterations (e.g. CMA-ME's distribution)."""

from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from toop_engine_topology_optimizer.dc.algorithm_loader.context import Context
from toop_engine_topology_optimizer.dc.ga_helpers import MixingEmitterState, TrackingMixingEmitter
from toop_engine_topology_optimizer.dc.genetic_functions.genotype import Genotype
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import DiscreteMapElitesRepertoire


class MemoryState(MixingEmitterState):
    """ToOp's emitter state (the counters) plus the algorithm's memory."""

    memory: PyTree
    """Any pytree of fixed-shape arrays, e.g. a dict of jnp arrays"""


class EmitterWithMemory(TrackingMixingEmitter):
    """An emitter whose emit can read emitter_state.memory.

    Subclasses write init_memory, update_memory and emit. The counters ToOp reports are kept as before.
    """

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx.mutate_fn, ctx.crossover_fn, 0.0, ctx.batch_size)
        self._ctx = ctx

    def init_memory(self, random_key: PRNGKeyArray) -> PyTree:  # noqa: ARG002
        """Return the memory at the start of the run."""
        return None

    def update_memory(
        self,
        memory: PyTree,
        genotypes: Genotype,  # noqa: ARG002
        fitnesses: Float[Array, " n"],  # noqa: ARG002
        repertoire: DiscreteMapElitesRepertoire,  # noqa: ARG002
    ) -> PyTree:
        """Return the new memory, given the topologies just scored and their fitnesses (higher is better)."""
        return memory

    def init(self, random_key: PRNGKeyArray, init_genotypes: Genotype) -> tuple[MemoryState, PRNGKeyArray]:
        """Start the counters at zero and set up the memory."""
        counters, random_key = super().init(random_key, init_genotypes)
        return MemoryState(
            total_branch_combis=counters.total_branch_combis,
            total_inj_combis=counters.total_inj_combis,
            total_num_splits=counters.total_num_splits,
            memory=self.init_memory(random_key),
        ), random_key

    def state_update(
        self,
        emitter_state: MemoryState,
        repertoire: DiscreteMapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Float[Array, " n"],
        descriptors: Array,
        extra_scores: dict,
    ) -> MemoryState:
        """Update the counters as before, then the memory."""
        counters = super().state_update(emitter_state, repertoire, genotypes, fitnesses, descriptors, extra_scores)
        return MemoryState(
            total_branch_combis=counters.total_branch_combis,
            total_inj_combis=counters.total_inj_combis,
            total_num_splits=counters.total_num_splits,
            memory=self.update_memory(emitter_state.memory, genotypes, fitnesses, repertoire),
        )
