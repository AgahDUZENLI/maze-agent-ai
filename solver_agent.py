"""
Compatibility agent entry points for the local solver workflow.

The canonical implementation lives in `maze_solver.py`. This module keeps the
solver-facing wrapper available without replacing the upstream `agent.py`.
"""

from maze_reader import DEFAULT_WALLS_PATH, find_start_goal, load_maze
from maze_solver import Agent as PartialObservationAgent
from maze_solver import fresh_knowledge_state


class HybridAgent(PartialObservationAgent):
    """
    Wrapper around the current solver agent with upstream-style construction.
    """

    def __init__(self, start=None, goal=None, knowledge=None, walls_path=DEFAULT_WALLS_PATH):
        if start is None or goal is None:
            _, h_walls, _ = load_maze(str(walls_path))
            default_start, default_goal = find_start_goal(h_walls)
            start = default_start if start is None else start
            goal = default_goal if goal is None else goal

        if knowledge is None:
            knowledge = fresh_knowledge_state()

        super().__init__(start, goal, knowledge)

    def reset_episode(self, start_pos=None):
        if start_pos is not None:
            self.start = tuple(start_pos)
            self.knowledge["cell_types"][self.start] = "start"
        super().reset_episode()

    def get_metrics(self):
        known_cells = dict(self.knowledge["cell_types"])
        wall_count = sum(len(directions) for directions in self.knowledge["walls"].values())
        teleport_cells = sum(1 for value in known_cells.values() if value == "teleport")
        confusion_cells = sum(1 for value in known_cells.values() if value == "confusion")
        lethal_cells = sum(1 for value in known_cells.values() if value == "lethal")

        return {
            "phase": "frontier-bfs",
            "goal_found": known_cells.get(self.goal) == "goal",
            "goal_pos": self.goal if known_cells.get(self.goal) == "goal" else None,
            "unique_cells": len(known_cells),
            "walls_mapped": wall_count,
            "teleports_mapped": teleport_cells,
            "confuse_mapped": confusion_cells,
            "deaths_mapped": lethal_cells,
            "wall_hits": self.wall_hits,
            "deaths": self.deaths,
            "teleports": self.teleports,
            "confusion_entries": self.confusion_entries,
        }

    def print_metrics(self):
        metrics = self.get_metrics()
        print("\nCompatibility agent metrics")
        print(f"  phase              : {metrics['phase']}")
        print(f"  goal_found         : {metrics['goal_found']}")
        print(f"  unique_cells       : {metrics['unique_cells']}")
        print(f"  walls_mapped       : {metrics['walls_mapped']}")
        print(f"  teleports_mapped   : {metrics['teleports_mapped']}")
        print(f"  confusion_cells    : {metrics['confuse_mapped']}")
        print(f"  lethal_cells       : {metrics['deaths_mapped']}")
        print(f"  wall_hits          : {metrics['wall_hits']}")
        print(f"  deaths             : {metrics['deaths']}")


Agent = PartialObservationAgent

__all__ = ["Agent", "HybridAgent", "PartialObservationAgent", "fresh_knowledge_state"]
