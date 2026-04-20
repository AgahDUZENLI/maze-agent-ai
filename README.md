# Blind Maze Agent

This repository contains the current spec-aligned maze solver and replay tools.

The project now relies on the maze assets under `TestMazes/`. The default asset pair used by the tools is `TestMazes/maze-alpha/`.

The active runtime is:

- `environment.py`: defines `Action`, `TurnResult`, and `MazeEnvironment.step(actions)`
- `maze_solver.py`: implements the agent and evaluation loop
- `maze_reader.py`: loads walls and hazards from the PNG assets
- `maze_printer.py`: renders the main annotated maze outputs
- `maze_visualizer.py`: renders replay GIFs from saved run data

Compatibility entry points restored from the upstream repository layout:

- `agent.py`: compatibility wrapper around the current solver agent
- `train.py`: upstream-style runner that still uses the current environment and knowledge format
- `visualizer.py`: legacy static episode snapshot renderer; replay GIFs still belong to `maze_visualizer.py`

The solver updates its knowledge only from turn feedback. It does not inspect the maze state directly while solving.

## Search Strategy

The current agent uses a partial-observation frontier strategy:

- greedy probing for unknown adjacent directions
- BFS over the learned safe graph when routing to the goal
- BFS over the learned safe graph when routing to the nearest reachable frontier

In short: it explores locally with a simple heuristic, and it navigates known territory with BFS.

## Current Behavior

- The agent starts with only the start and goal coordinates.
- It sends actions through `MazeEnvironment.step(...)`.
- The active solver intentionally uses one move per turn so every wall hit, death, teleport, and confusion event can be attributed exactly.
- Run knowledge is saved to JSON and reused across episodes for the same maze.

Saved knowledge includes:

- discovered exits with visit/death counts and destinations
- confirmed wall directions
- inferred cell types such as `safe`, `confusion`, `teleport`, `goal`, and `lethal`
- per-step replay `events` so the visualizer can reconstruct run semantics accurately

## Metrics

`maze_solver.py` reports:

- success rate
- average turns
- average path length
- average deaths
- average wall hits
- average exploration efficiency

Exploration efficiency is:

```text
cells_explored / total_cells_visited
```

## Running The Solver

Default maze:

```bash
python3 maze_solver.py
```

Single fresh run on each test maze:

```bash
python3 maze_solver.py --walls "TestMazes/maze-alpha/MAZE_0.png" --hazards "TestMazes/maze-alpha/MAZE_1.png" --episodes 1 --reset-knowledge --knowledge "results/maze-alpha.json" --no-render
python3 maze_solver.py --walls "TestMazes/maze-beta/MAZE_0.png" --hazards "TestMazes/maze-beta/MAZE_1.png" --episodes 1 --reset-knowledge --knowledge "results/maze-beta.json" --no-render
python3 maze_solver.py --walls "TestMazes/maze-gamma/MAZE_0.png" --hazards "TestMazes/maze-gamma/MAZE_1.png" --episodes 1 --reset-knowledge --knowledge "results/maze-gamma.json" --no-render
```

Multiple episodes:

```bash
python3 maze_solver.py --episodes 5
```

Optional explicit teleport mapping:

```bash
python3 maze_solver.py --teleports teleports.json
```

Expected JSON format:

```json
{
  "12,7": [40, 11],
  "40,11": [12, 7]
}
```

## Rendering Annotated Maze Outputs

`maze_printer.py` is the primary static visualization tool for the project.

Default maze:

```bash
python3 maze_printer.py
```

Per-test-maze annotated outputs:

```bash
python3 maze_printer.py --walls "TestMazes/maze-alpha/MAZE_0.png" --hazards "TestMazes/maze-alpha/MAZE_1.png"
python3 maze_printer.py --walls "TestMazes/maze-beta/MAZE_0.png" --hazards "TestMazes/maze-beta/MAZE_1.png"
python3 maze_printer.py --walls "TestMazes/maze-gamma/MAZE_0.png" --hazards "TestMazes/maze-gamma/MAZE_1.png"
```

By default the printer writes into:

```text
results/printer/<maze-name>/
```

It generates:

- `annotated_maze.png`
- `annotated_maze_legend.png`
- `maze_turn_0.png ... maze_turn_N.png`

`maze_visualizer.py` is kept for replay output only.

## Rendering Replays

Generate GIFs from saved result JSONs:

```bash
python3 maze_visualizer.py --walls "TestMazes/maze-alpha/MAZE_0.png" --hazards "TestMazes/maze-alpha/MAZE_1.png" --knowledge "results/maze-alpha.json" --output "results/visualizer/maze-alpha.gif" --no-latest-alias
python3 maze_visualizer.py --walls "TestMazes/maze-beta/MAZE_0.png" --hazards "TestMazes/maze-beta/MAZE_1.png" --knowledge "results/maze-beta.json" --output "results/visualizer/maze-beta.gif" --no-latest-alias
python3 maze_visualizer.py --walls "TestMazes/maze-gamma/MAZE_0.png" --hazards "TestMazes/maze-gamma/MAZE_1.png" --knowledge "results/maze-gamma.json" --output "results/visualizer/maze-gamma.gif" --no-latest-alias
```

Replay behavior:

- By default, the visualizer rotates fire for display using the project turn cadence.
- `--static-fire` is a display override that shows the saved run hazards instead.
- The HUD and CLI output show both the run hazard model and the display hazard model.

## Current Maze Results

Fresh single-episode runs on the normalized test mazes:

| Maze | Result | Turns | Steps | Deaths | Wall Hits |
|------|--------|------:|------:|-------:|----------:|
| `maze-alpha` | Success | 8398 | 4513 | 6 | 3885 |
| `maze-beta` | Failed | 1222 | 633 | 3 | 589 |
| `maze-gamma` | Failed | 10228 | 5933 | 17 | 4295 |

## Assumptions And Limits

Two important limits still come from the provided PNG assets:

1. The images do not encode arbitrary teleport graph metadata. Without an explicit JSON mapping, teleports fall back to color-paired behavior.
2. The active solver uses the static/spec-style hazard interpretation. The visualizer rotates fire for presentation by default, but that does not change the saved run semantics.

## AI Usage Disclosure

AI assistance was used during development to inspect the repository, compare the implementation against the assignment specification, and help draft/refactor code and documentation. Final code changes were reviewed and executed locally in this repository.
