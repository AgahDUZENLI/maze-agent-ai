"""
Compatibility training entry point for the local solver workflow.

This keeps the solver-driven training path available without replacing the
upstream `train.py`.
"""

import argparse
from pathlib import Path

from maze_reader import DEFAULT_HAZARDS_PATH, DEFAULT_WALLS_PATH, load_maze
from maze_solver import (
    KNOWLEDGE_FORMAT_VERSION,
    evaluate_agent,
    fresh_knowledge_state,
    load_knowledge,
    reconstruct_shortest_known_path,
    render_path,
    resolve_project_path,
    save_knowledge,
)
from solver_agent import HybridAgent
from solver_environment import MazeEnvironment
from visualizer import MazeVisualizer

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


def maze_label_from_paths(walls_path, hazards_path):
    walls_path = Path(walls_path)
    hazards_path = Path(hazards_path)
    if walls_path.parent != BASE_DIR:
        return walls_path.parent.name
    if hazards_path.parent != BASE_DIR:
        return hazards_path.parent.name
    return walls_path.stem.lower()


def default_knowledge_path(walls_path, hazards_path):
    return RESULTS_DIR / f"{maze_label_from_paths(walls_path, hazards_path)}.json"


def default_viz_dir(walls_path, hazards_path):
    return RESULTS_DIR / "legacy-viz" / maze_label_from_paths(walls_path, hazards_path)


def death_cells_from_record(record):
    death_cells = []
    for event in record.get("events", []):
        if event.get("kind") == "death":
            row, col = event["position"]
            death_cells.append((row, col))
    return death_cells


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the solver compatibility training workflow."
    )
    parser.add_argument(
        "--walls",
        default=DEFAULT_WALLS_PATH,
        help="Wall image path, relative to the project root by default.",
    )
    parser.add_argument(
        "--hazards",
        default=DEFAULT_HAZARDS_PATH,
        help="Hazard image path, relative to the project root by default.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--max-turns", type=int, default=20000, help="Hard turn cap per episode.")
    parser.add_argument(
        "--knowledge",
        default=None,
        help="Knowledge JSON path. Defaults to results/<maze-name>.json.",
    )
    parser.add_argument(
        "--viz-dir",
        default=None,
        help="Directory for legacy static episode snapshots.",
    )
    parser.add_argument(
        "--reset-knowledge",
        action="store_true",
        help="Ignore any existing knowledge file and start fresh.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip path renders and legacy episode snapshots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    walls_path = resolve_project_path(args.walls)
    hazards_path = resolve_project_path(args.hazards)
    knowledge_path = (
        resolve_project_path(args.knowledge)
        if args.knowledge
        else default_knowledge_path(walls_path, hazards_path)
    )
    viz_dir = (
        resolve_project_path(args.viz_dir)
        if args.viz_dir
        else default_viz_dir(walls_path, hazards_path)
    )

    print("=" * 55)
    print("SOLVER COMPATIBILITY TRAIN RUN")
    print("=" * 55)

    knowledge = fresh_knowledge_state() if args.reset_knowledge else load_knowledge(knowledge_path)
    environment = MazeEnvironment(
        maze_id=maze_label_from_paths(walls_path, hazards_path),
        walls_path=walls_path,
        hazards_path=hazards_path,
    )
    agent = HybridAgent(start=environment.start, goal=environment.goal, knowledge=knowledge)

    run_records, evaluation = evaluate_agent(
        environment=environment,
        agent=agent,
        episodes=args.episodes,
        max_turns=args.max_turns,
        starting_run=knowledge["runs"],
    )

    metadata = {
        "solver_state_version": KNOWLEDGE_FORMAT_VERSION,
        "solver_mode": "spec-partial-observation",
        "hazard_model": "static",
        "teleport_model": "asset-paired-fallback",
        "walls_path": str(walls_path),
        "hazards_path": str(hazards_path),
        "entry_point": "solver_train.py",
    }
    move_history = knowledge["move_history"] + run_records
    save_knowledge(knowledge_path, knowledge, move_history, evaluation, metadata)

    print(f"\nKnowledge saved -> {knowledge_path}")
    print(f"Episodes        : {evaluation['episodes']}")
    print(f"Success rate    : {evaluation['success_rate']:.3f}")
    print(f"Avg turns       : {evaluation['avg_turns']:.2f}")
    print(f"Avg path length : {evaluation['avg_path_length']:.2f}")
    print(f"Avg deaths      : {evaluation['avg_deaths']:.2f}")
    print(f"Avg wall hits   : {evaluation['avg_wall_hits']:.2f}")

    if args.no_render or not run_records:
        return

    _, h_walls, v_walls = load_maze(str(walls_path))
    visualizer = MazeVisualizer(str(walls_path))

    for record in run_records:
        full_path = [environment.start] + [tuple(cell) for cell in record["moves"]]
        render_path(
            h_walls,
            v_walls,
            full_path,
            environment.start,
            environment.goal,
            record["run"],
            optimized=False,
        )

        if record["success"]:
            optimized_path = reconstruct_shortest_known_path(
                knowledge["cell_exits"],
                environment.start,
                environment.goal,
            )
            if optimized_path:
                render_path(
                    h_walls,
                    v_walls,
                    optimized_path,
                    environment.start,
                    environment.goal,
                    record["run"],
                    optimized=True,
                )

        visualizer.save_episode(
            episode_num=record["run"],
            agent=agent,
            env=environment,
            path_taken=full_path,
            death_cells=death_cells_from_record(record),
            events=record.get("events"),
            output_dir=viz_dir,
        )

    print(f"Legacy episode snapshots -> {viz_dir}")


if __name__ == "__main__":
    main()
