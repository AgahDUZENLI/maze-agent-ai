import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from maze_reader import (
    DEFAULT_HAZARDS_PATH,
    DEFAULT_WALLS_PATH,
    Hazard,
    can_move,
    get_goal,
    get_start,
    get_teleport_points,
    load_hazards,
    load_maze,
)


class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4


ACTION_TO_DIRECTION = {
    Action.MOVE_UP: "up",
    Action.MOVE_DOWN: "down",
    Action.MOVE_LEFT: "left",
    Action.MOVE_RIGHT: "right",
}

INVERTED_ACTION = {
    Action.MOVE_UP: Action.MOVE_DOWN,
    Action.MOVE_DOWN: Action.MOVE_UP,
    Action.MOVE_LEFT: Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
    Action.WAIT: Action.WAIT,
}

MOVE_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass
class TurnResult:
    wall_hits: int = 0
    current_position: tuple[int, int] = (0, 0)
    is_dead: bool = False
    is_confused: bool = False
    is_goal_reached: bool = False
    teleported: bool = False
    actions_executed: int = 0


class MazeEnvironment:
    """
    Canonical spec-faithful environment used by every project entry point.

    The assignment PDF requires static death pits and one-way deterministic
    teleports. The PNG assets do not encode an explicit destination graph, so the
    default fallback pairs same-color teleport cells in sorted order and maps the
    first cell of each pair to the second.
    """

    def __init__(
        self,
        maze_id="training",
        walls_path=None,
        hazards_path=None,
        teleport_map=None,
        teleport_map_path=None,
    ):
        # Backward-compatible positional form used by the legacy train.py:
        # MazeEnvironment("MAZE_0.png", "MAZE_1.png")
        if walls_path is not None and hazards_path is None and maze_id not in {"training", "testing"}:
            hazards_path = walls_path
            walls_path = maze_id
            maze_id = Path(str(walls_path)).stem

        if walls_path is None:
            walls_path = DEFAULT_WALLS_PATH
        if hazards_path is None:
            hazards_path = DEFAULT_HAZARDS_PATH

        self.maze_id = maze_id
        self.walls_path = str(walls_path)
        self.hazards_path = str(hazards_path)
        _, self.h_walls, self.v_walls = load_maze(str(walls_path))
        self.hazards = load_hazards(str(hazards_path))
        self.start = get_start(self.h_walls)
        self.goal = get_goal(self.h_walls)
        self.teleport_map = self._build_teleport_map(
            self.hazards,
            teleport_map=teleport_map,
            teleport_map_path=teleport_map_path or self._default_teleport_map_path(),
        )

        self.position = self.start
        self.pending_respawn = False
        self.confused_next_turn = False
        self.turns_taken = 0
        self.deaths = 0
        self.confused = 0
        self.goal_reached = False
        self.cells_explored = {self.start}
        self.total_cells_visited = 1
        self.path_length = 0

    def _parse_cell(self, value):
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("(") and text.endswith(")"):
                row_str, col_str = text[1:-1].split(",")
            else:
                row_str, col_str = text.split(",")
            return (int(row_str.strip()), int(col_str.strip()))
        return (int(value[0]), int(value[1]))

    def _normalize_action(self, action):
        if isinstance(action, Action):
            return action
        return Action(int(action))

    def _default_teleport_map_path(self):
        hazards_path = Path(self.hazards_path)
        if not hazards_path.is_absolute():
            hazards_path = Path(__file__).resolve().parent / hazards_path
        candidate = hazards_path.parent / "teleports.json"
        return candidate if candidate.exists() else None

    def _load_explicit_teleport_map(self, teleport_map=None, teleport_map_path=None):
        raw_mapping = teleport_map
        if raw_mapping is None and teleport_map_path is not None:
            with open(Path(teleport_map_path), encoding="utf-8") as handle:
                raw_mapping = json.load(handle)

        if raw_mapping is None:
            return None

        mapping = {}
        for source, destination in raw_mapping.items():
            mapping[self._parse_cell(source)] = self._parse_cell(destination)
        return mapping

    def _build_teleport_map(self, hazards, teleport_map=None, teleport_map_path=None):
        explicit_map = self._load_explicit_teleport_map(
            teleport_map=teleport_map,
            teleport_map_path=teleport_map_path,
        )
        if explicit_map is not None:
            return explicit_map

        mapping = {}
        for cells in get_teleport_points(hazards).values():
            # The assets only expose pad colors, not destination metadata. Pair
            # pads in sorted order so the fallback remains deterministic and
            # one-way instead of the previous bidirectional interpretation.
            for index in range(0, len(cells) - 1, 2):
                mapping[cells[index]] = cells[index + 1]
        return mapping

    def reset(self):
        self.position = self.start
        self.pending_respawn = False
        self.confused_next_turn = False
        self.turns_taken = 0
        self.deaths = 0
        self.confused = 0
        self.goal_reached = False
        self.cells_explored = {self.start}
        self.total_cells_visited = 1
        self.path_length = 0
        return self.start

    def _apply_teleport(self, position):
        destination = self.teleport_map.get(position)
        if destination is None:
            return position, False
        self.cells_explored.add(destination)
        self.total_cells_visited += 1
        return destination, True

    def step(self, actions):
        if not actions or len(actions) > 5:
            raise ValueError("actions must contain between 1 and 5 items")

        if self.goal_reached:
            raise RuntimeError("episode already ended; call reset() before stepping again")

        if self.pending_respawn:
            self.position = self.start
            self.pending_respawn = False

        result = TurnResult(current_position=self.position)
        controls_inverted = self.confused_next_turn
        self.confused_next_turn = False
        stepped_on_confusion = False

        for raw_action in actions:
            if result.is_dead or result.is_goal_reached:
                break

            action = self._normalize_action(raw_action)
            executed_action = INVERTED_ACTION[action] if controls_inverted else action

            if executed_action == Action.WAIT:
                result.actions_executed += 1
                continue

            direction = ACTION_TO_DIRECTION[executed_action]
            if not can_move(self.position[0], self.position[1], direction, self.h_walls, self.v_walls):
                result.wall_hits += 1
                result.actions_executed += 1
                continue

            dr, dc = MOVE_DELTAS[direction]
            self.position = (self.position[0] + dr, self.position[1] + dc)
            self.cells_explored.add(self.position)
            self.total_cells_visited += 1
            self.path_length += 1
            result.actions_executed += 1

            hazard = self.hazards.get(self.position)
            if hazard == Hazard.FIRE:
                result.is_dead = True
                result.current_position = self.position
                self.deaths += 1
                self.pending_respawn = True
                break

            self.position, did_teleport = self._apply_teleport(self.position)
            if did_teleport:
                result.teleported = True

            if self.hazards.get(self.position) == Hazard.CONFUSION:
                stepped_on_confusion = True
                controls_inverted = True
                self.confused += 1

            result.current_position = self.position

            if self.position == self.goal:
                result.is_goal_reached = True
                self.goal_reached = True
                break

        if not result.is_dead:
            result.current_position = self.position

        result.is_confused = stepped_on_confusion
        if stepped_on_confusion:
            self.confused_next_turn = True

        self.turns_taken += 1
        return result

    def get_episode_stats(self):
        return {
            "turns_taken": self.turns_taken,
            "deaths": self.deaths,
            "confused": self.confused,
            "cells_explored": len(self.cells_explored),
            "goal_reached": self.goal_reached,
            "path_length": self.path_length,
            "total_cells_visited": self.total_cells_visited,
        }
