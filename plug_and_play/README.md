# Plug and play

Pick a search algorithm, run it on a grid, compare the results. Everything you need is in this folder; the rest of the repository is ToOp itself.

```
plug_and_play/
  algorithms/   one file per algorithm
  run.py        choose grids and algorithms, then run
  results/      one folder per run of run.py
```

## Set up (once)

```
git clone https://github.com/lakg90/ToOpProject.git
cd ToOpProject
git checkout plugins
uv sync
```

## 1. Pick algorithms

| Name | What it is |
|---|---|
| `map_elites` | ToOp's own algorithm |
| `random_search` | Baseline: random plans every step |
| `local_search` | Baseline: small changes to the best plan so far |
| `surrogate_map_elites` | Starter: runs, but the model is still to be written |
| `cma_me` | Starter: runs, but the method is still to be written |
| `multi_objective` | Starter for NSGA-II and MOME: picks parents from the Pareto front; the full method is still to be written |

Starters print a warning when they run, so their results are not mistaken for the real method.

## 2. Run them on grids

Open `run.py` and edit the settings at the top:

```python
GRIDS = ["path/to/grid_a", "path/to/grid_b"]
ALGORITHMS = ["map_elites", "random_search"]
SEEDS = 3
SECONDS = 60
```

A grid is a folder prepared by ToOp's preprocessing (it holds `static_information.hdf5`, or has a `toop/` subfolder that does). Then run:

```
uv run python plug_and_play/run.py
```

Each run is ToOp's DC search with that algorithm, then ToOp's AC check on the 10 best plans.

## 3. Look at the results

Each time you run `run.py` it makes a new folder in `results/`:

| File | What is in it |
|---|---|
| `runs.csv` | One row per grid, algorithm and seed: plans sent to AC, plans that passed, best AC overload, time |
| `candidates.csv` | One row per plan sent to AC: DC and AC overload, passed or not, and why it was rejected |
| `settings.json` | The settings and code version, so a result can be reproduced |

## Writing an algorithm

1. Copy `algorithms/template.py` to `algorithms/my_algorithm.py`. The file name is the algorithm's name.
2. Write the `emit` function: given the plans kept so far, return the next batch of plans to try.
3. Put `"my_algorithm"` in `ALGORITHMS` in `run.py` and run it.

To finish a starter, open its file and fill in the parts marked `TODO`. If the algorithm needs to remember something between steps (such as CMA-ME's probabilities), copy `cma_me.py` instead of the template.

What an algorithm can use:

- `self._ctx`: batch size, the available splits, ToOp's mutation and crossover, and any settings passed as `algorithm_params`
- `repertoire`: the plans kept so far, their scores (`fitnesses`, higher is better) and every metric for each plan (`extra_scores`)

`emit` runs inside `jax.jit`, so use `jnp` and `jax.random` (not `numpy` or `random`), no Python `if` on array values (use `jnp.where`), and always return exactly `batch_size` plans.

To check every algorithm in the folder still works:

```
uv run pytest packages/topology_optimizer_pkg/tests/dc/test_plug_and_play.py
```

## Limitation

This swaps how plans are suggested, not the whole search. Methods that suggest batches of plans fit. MCTS and QD-RL build plans step by step or need training, so they need their own loop.
