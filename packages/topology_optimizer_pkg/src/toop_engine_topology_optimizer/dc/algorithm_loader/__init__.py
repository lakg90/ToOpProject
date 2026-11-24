"""Load the search algorithm (emitter) named in ga_config.algorithm.

Algorithms are the files in plug_and_play/algorithms at the top of the repository: the file name is the
algorithm name and the file must have a build(ctx) function. map_elites is always available, so ToOp
behaves as before if that folder is missing.
"""

import importlib.util
import os
from pathlib import Path

from qdax.core.emitters.emitter import Emitter
from toop_engine_topology_optimizer.dc.algorithm_loader.context import Context
from toop_engine_topology_optimizer.dc.algorithm_loader.memory import EmitterWithMemory, MemoryState
from toop_engine_topology_optimizer.dc.ga_helpers import TrackingMixingEmitter

__all__ = ["Context", "EmitterWithMemory", "MemoryState", "algorithms_folder", "available", "build_algorithm"]


def algorithms_folder() -> Path:
    """Return the folder with one file per algorithm. Can be changed with the TOOP_ALGORITHMS_DIR variable."""
    if "TOOP_ALGORITHMS_DIR" in os.environ:
        return Path(os.environ["TOOP_ALGORITHMS_DIR"])
    return Path(__file__).resolve().parents[6] / "plug_and_play" / "algorithms"


def available() -> list[str]:
    """Return the names of all algorithms that can be chosen."""
    folder = algorithms_folder()
    names = {p.stem for p in folder.glob("*.py") if not p.stem.startswith("_") and p.stem != "template"}
    return sorted(names | {"map_elites"})


def build_algorithm(name: str, ctx: Context) -> Emitter:
    """Build the emitter for the named algorithm.

    Parameters
    ----------
    name : str
        The value of ga_config.algorithm, the name of a file in the algorithms folder
    ctx : Context
        What the algorithm needs to know about the problem

    Returns
    -------
    Emitter
        The emitter that suggests new topologies

    Raises
    ------
    ValueError
        If no algorithm with that name exists
    """
    path = algorithms_folder() / f"{name}.py"
    if path.exists():
        spec = importlib.util.spec_from_file_location(f"plug_and_play_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build(ctx)
    if name == "map_elites":
        return TrackingMixingEmitter(ctx.mutate_fn, ctx.crossover_fn, ctx.crossover_mutation_ratio, ctx.batch_size)
    raise ValueError(f"Unknown algorithm {name!r}. Available: {available()}")
