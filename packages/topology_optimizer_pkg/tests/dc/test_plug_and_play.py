import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest
from fsspec.implementations.local import LocalFileSystem
from toop_engine_topology_optimizer.dc.algorithm_loader import algorithms_folder, available
from toop_engine_topology_optimizer.dc.main import CLIArgs, main
from toop_engine_topology_optimizer.dc.worker.optimizer import initialize_optimization, run_epoch
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    DCOptimizerParameters,
    LoadflowSolverParameters,
)


def run_short(static_information_file: Path, algorithm: str, seed: int = 0) -> tuple:
    jax.config.update("jax_enable_x64", True)
    optimizer_data, _, _ = initialize_optimization(
        params=DCOptimizerParameters(
            ga_config=BatchedMEParameters(algorithm=algorithm, random_seed=seed, iterations_per_epoch=5),
            loadflow_solver_config=LoadflowSolverParameters(batch_size=8),
        ),
        optimization_id="test",
        static_information_files=(str(static_information_file),),
        processed_gridfile_fs=LocalFileSystem(),
    )
    optimizer_data = run_epoch(optimizer_data)
    return optimizer_data.jax_data.repertoire, optimizer_data.jax_data.emitter_state


def test_algorithms_folder_found() -> None:
    assert algorithms_folder().is_dir()
    assert {"map_elites", "random_search", "local_search", "cma_me"} <= set(available())


@pytest.mark.parametrize("algorithm", available())
def test_algorithm_runs(static_information_file: Path, algorithm: str) -> None:
    repertoire, emitter_state = run_short(static_information_file, algorithm)
    assert np.isfinite(np.asarray(repertoire.fitnesses)).any()
    # 6 batches of 8: one at initialisation, then 5 iterations
    assert int(emitter_state.total_branch_combis) == 48


@pytest.mark.parametrize("algorithm", available())
def test_algorithm_same_seed_same_result(static_information_file: Path, algorithm: str) -> None:
    first, _ = run_short(static_information_file, algorithm, seed=3)
    second, _ = run_short(static_information_file, algorithm, seed=3)
    np.testing.assert_array_equal(first.genotypes.action_index, second.genotypes.action_index)
    np.testing.assert_array_equal(first.fitnesses, second.fitnesses)


def test_map_elites_unchanged_by_file(static_information_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from_file, _ = run_short(static_information_file, "map_elites", seed=1)
    monkeypatch.setenv("TOOP_ALGORITHMS_DIR", "/folder/that/does/not/exist")
    built_in, _ = run_short(static_information_file, "map_elites", seed=1)
    np.testing.assert_array_equal(from_file.fitnesses, built_in.fitnesses)


def test_unknown_algorithm(static_information_file: Path) -> None:
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_short(static_information_file, "does_not_exist")


def test_memory_is_updated(static_information_file: Path) -> None:
    _, emitter_state = run_short(static_information_file, "cma_me")
    # once at initialisation, then once per iteration
    assert int(emitter_state.memory["n_updates"]) == 6
    assert np.isfinite(float(emitter_state.memory["best_fitness"]))


def test_first_front() -> None:
    spec = importlib.util.spec_from_file_location("multi_objective", algorithms_folder() / "multi_objective.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    objectives = jax.numpy.array([[1.0, 5.0], [2.0, 2.0], [3.0, 3.0], [5.0, 1.0], [0.0, 0.0]])
    valid = jax.numpy.array([True, True, True, True, False])
    # [3, 3] is beaten by [2, 2]; the last row is ignored because it is not valid
    np.testing.assert_array_equal(module.first_front(objectives, valid), [True, True, False, True, False])


@pytest.mark.parametrize("algorithm", available())
def test_algorithm_full_main(tmp_path: Path, static_information_file: Path, algorithm: str) -> None:
    """The whole DC optimiser, including writing res.json, as run_pipeline and plug_and_play/run.py use it."""
    args = CLIArgs(
        ga_config=BatchedMEParameters(runtime_seconds=5, iterations_per_epoch=2, algorithm=algorithm),
        stats_dir=str(tmp_path / "res_dir"),
        tensorboard_dir=str(tmp_path / "tensorboard"),
        fixed_files=(str(static_information_file),),
    )
    result = main(args, processed_gridfile_fs=LocalFileSystem())
    assert result["total_branch_combis"] > 0
    assert (tmp_path / "res_dir" / "res.json").exists()
