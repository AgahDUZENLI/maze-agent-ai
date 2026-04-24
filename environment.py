import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

from maze_reader import (
    load_maze,
    load_hazards,
    find_start_goal,
    update_fire_in_hazards,
    init_fire_groups,
    get_teleport_pairs,
    can_move,
    Hazard,
    GRID,
)

# ── SAFE FALLBACK (push hazards not implemented yet) ────────────────────────
def is_push_hazard(hz):
    return False

def push_direction_for_hazard(hz):
    return (0, 0)

# ── Action constants ────────────────────────────────────────────────────────
ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3
ACTION_WAIT  = 4

VALID_ACTIONS = {ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_WAIT}

DELTAS = {
    ACTION_UP:    (-1,  0),
    ACTION_DOWN:  ( 1,  0),
    ACTION_LEFT:  ( 0, -1),
    ACTION_RIGHT: ( 0,  1),
}

DIRECTION_NAMES = {
    ACTION_UP:    "up",
    ACTION_DOWN:  "down",
    ACTION_LEFT:  "left",
    ACTION_RIGHT: "right",
}

INVERT_ACTION = {
    ACTION_UP:    ACTION_DOWN,
    ACTION_DOWN:  ACTION_UP,
    ACTION_LEFT:  ACTION_RIGHT,
    ACTION_RIGHT: ACTION_LEFT,
}

# ── TurnResult ──────────────────────────────────────────────────────────────
class TurnResult:
    def __init__(self):
        self.wall_hits = 0
        self.current_position = (0, 0)
        self.is_dead = False
        self.is_confused = False
        self.is_goal_reached = False
        self.teleported = False
        self.was_forced = False
        self.forced_direction = None
        self.actions_executed = 0

    def __repr__(self):
        return (
            f"TurnResult(pos={self.current_position}, "
            f"dead={self.is_dead}, goal={self.is_goal_reached}, "
            f"wall_hits={self.wall_hits}, teleported={self.teleported}, "
            f"forced={self.was_forced}, forced_dir={self.forced_direction}, "
            f"confused={self.is_confused}, actions={self.actions_executed})"
        )

# ── MazeEnvironment ─────────────────────────────────────────────────────────
class MazeEnvironment:

    def __init__(self, maze_name="alpha"):

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        MAZE_DIR  = os.path.join(BASE_DIR, "TestMazes", f"maze-{maze_name}")
        MAZE_PATH = os.path.join(MAZE_DIR, "MAZE_0.png")
        HAZ_PATH  = os.path.join(MAZE_DIR, "MAZE_1.png")
        self.maze_name = maze_name
        self.maze_path = MAZE_PATH
        self.hazard_path = HAZ_PATH

        print(f"Loading maze walls  : {MAZE_PATH}")
        self.image, self.h_walls, self.v_walls = load_maze(MAZE_PATH)

        print(f"Loading hazards     : {HAZ_PATH}")
        self.base_hazards = load_hazards(HAZ_PATH)

        self.hazards = dict(self.base_hazards)
        self.fire_pivots = init_fire_groups(self.hazards)

        self.start, self.goal = find_start_goal(self.h_walls)
        print(f"Start               : {self.start}")
        print(f"Goal                : {self.goal}")

        self.teleport_map = self._build_teleport_map(self.hazards)
        print(f"Teleporter pairs    : {len(self.teleport_map)//2}")

        from collections import Counter
        counts = Counter(self.hazards.values())

        print(f"Fire cells          : {counts.get(Hazard.FIRE, 0)}")
        print(f"Confusion traps     : {counts.get(Hazard.CONFUSION, 0)}")

        tp_count = sum(
            counts.get(h, 0)
            for h in [Hazard.TP_GREEN, Hazard.TP_YELLOW, Hazard.TP_PURPLE, Hazard.TP_RED]
        )
        print(f"Teleporters (cells) : {tp_count}\n")

        self.agent_pos = self.start
        self.is_confused = False
        self.turn_count = 0
        self.death_count = 0
        self.confused_count = 0
        self.cells_visited = []
        self.goal_reached = False
        self.episode_number = 0
        self.replay_events = []

    # ────────────────────────────────────────────────────────────────────────
    def _build_teleport_map(self, hazards):
        teleport_map = {}
        pairs = get_teleport_pairs(hazards)

        for pair in pairs.values():
            if len(pair) == 2:
                a, b = pair
                teleport_map[a] = b
                teleport_map[b] = a

        return teleport_map

    # ────────────────────────────────────────────────────────────────────────
    def reset(self):
        self.hazards = dict(self.base_hazards)
        self.fire_pivots = init_fire_groups(self.hazards)
        self.teleport_map = self._build_teleport_map(self.hazards)

        self.agent_pos = self.start
        self.is_confused = False
        self.turn_count = 0
        self.death_count = 0
        self.confused_count = 0
        self.cells_visited = [self.start]
        self.goal_reached = False
        self.atomic_action_count = 0
        self.replay_events = []

        self.episode_number += 1
        return self.start

    # ────────────────────────────────────────────────────────────────────────
    def _tick_fire_clock(self):
        self.atomic_action_count += 1
        if self.atomic_action_count % 5 == 0:
            self.hazards, self.fire_pivots = update_fire_in_hazards(
                self.hazards, self.fire_pivots
            )
            return True
        return False

    # ────────────────────────────────────────────────────────────────────────
    def _record_event(
        self,
        turn_index,
        action_index,
        submitted_action,
        effective_action,
        position,
        wall_hit=False,
        teleported=False,
        confused=False,
        died=False,
        goal_reached=False,
        fire_rotated=False,
        respawn_to_start=False,
        forced=False,
        forced_direction=None,
    ):
        self.replay_events.append(
            {
                "turn_index": turn_index,
                "action_index": action_index,
                "atomic_action_count": self.atomic_action_count,
                "submitted_action": submitted_action,
                "effective_action": effective_action,
                "position": position,
                "wall_hit": wall_hit,
                "teleported": teleported,
                "confused": confused,
                "died": died,
                "goal_reached": goal_reached,
                "fire_rotated": fire_rotated,
                "respawn_to_start": respawn_to_start,
                "forced": forced,
                "forced_direction": forced_direction,
            }
        )

    # ────────────────────────────────────────────────────────────────────────
    def _apply_push_hazard(self, result, cell_hazard):
        if not is_push_hazard(cell_hazard):
            return cell_hazard

        forced_delta = push_direction_for_hazard(cell_hazard)

        for action, delta in DELTAS.items():
            if delta == forced_delta:
                result.was_forced = True
                result.forced_direction = action

                if can_move(*self.agent_pos, DIRECTION_NAMES[action], self.h_walls, self.v_walls):
                    dr, dc = delta
                    self.agent_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
                    self.cells_visited.append(self.agent_pos)

        return self.hazards.get(self.agent_pos)

    # ────────────────────────────────────────────────────────────────────────
    def step(self, actions: List[int]):
        if not actions:
            raise ValueError("Must submit at least 1 action per turn.")
        if len(actions) > 5:
            raise ValueError("Cannot submit more than 5 actions per turn.")

        result = TurnResult()
        self.turn_count += 1

        for i, submitted_action in enumerate(actions):
            if submitted_action not in VALID_ACTIONS:
                raise ValueError(f"Invalid action: {submitted_action}")

            result.actions_executed = i + 1

            if submitted_action == ACTION_WAIT:
                fire_rotated = self._tick_fire_clock()
                self._record_event(
                    turn_index=self.turn_count,
                    action_index=i + 1,
                    submitted_action=submitted_action,
                    effective_action=ACTION_WAIT,
                    position=self.agent_pos,
                    fire_rotated=fire_rotated,
                )
                continue

            effective_action = INVERT_ACTION[submitted_action] if self.is_confused else submitted_action

            if not can_move(*self.agent_pos, DIRECTION_NAMES[effective_action], self.h_walls, self.v_walls):
                result.wall_hits += 1
                fire_rotated = self._tick_fire_clock()
                self._record_event(
                    turn_index=self.turn_count,
                    action_index=i + 1,
                    submitted_action=submitted_action,
                    effective_action=effective_action,
                    position=self.agent_pos,
                    wall_hit=True,
                    fire_rotated=fire_rotated,
                )
                continue

            dr, dc = DELTAS[effective_action]
            self.agent_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            self.cells_visited.append(self.agent_pos)

            cell_hazard = self.hazards.get(self.agent_pos)

            # push (safe)
            cell_hazard = self._apply_push_hazard(result, cell_hazard)

            # teleport
            if self.agent_pos in self.teleport_map:
                self.agent_pos = self.teleport_map[self.agent_pos]
                self.cells_visited.append(self.agent_pos)
                result.teleported = True
                cell_hazard = self.hazards.get(self.agent_pos)

            # confusion
            if cell_hazard == Hazard.CONFUSION:
                self.is_confused = not self.is_confused
                result.is_confused = True
                self.confused_count += 1

            # death
            if cell_hazard == Hazard.FIRE:
                result.is_dead = True
                result.current_position = self.agent_pos
                self.death_count += 1
                fire_rotated = self._tick_fire_clock()
                self._record_event(
                    turn_index=self.turn_count,
                    action_index=i + 1,
                    submitted_action=submitted_action,
                    effective_action=effective_action,
                    position=result.current_position,
                    teleported=result.teleported,
                    confused=result.is_confused,
                    died=True,
                    fire_rotated=fire_rotated,
                    respawn_to_start=True,
                    forced=result.was_forced,
                    forced_direction=result.forced_direction,
                )
                self.agent_pos = self.start
                return result

            # goal
            if self.agent_pos == self.goal:
                result.is_goal_reached = True
                result.current_position = self.agent_pos
                self.goal_reached = True
                fire_rotated = self._tick_fire_clock()
                self._record_event(
                    turn_index=self.turn_count,
                    action_index=i + 1,
                    submitted_action=submitted_action,
                    effective_action=effective_action,
                    position=result.current_position,
                    teleported=result.teleported,
                    confused=result.is_confused,
                    goal_reached=True,
                    fire_rotated=fire_rotated,
                    forced=result.was_forced,
                    forced_direction=result.forced_direction,
                )
                return result

            fire_rotated = self._tick_fire_clock()
            self._record_event(
                turn_index=self.turn_count,
                action_index=i + 1,
                submitted_action=submitted_action,
                effective_action=effective_action,
                position=self.agent_pos,
                teleported=result.teleported,
                confused=result.is_confused,
                fire_rotated=fire_rotated,
                forced=result.was_forced,
                forced_direction=result.forced_direction,
            )

        result.current_position = self.agent_pos
        return result

    # ────────────────────────────────────────────────────────────────────────
    @property
    def turn(self):
        return self.atomic_action_count

    def get_fire_cells_at(self, phase: int) -> set:
        base_fire = {cell for cell, hz in self.base_hazards.items() if hz == Hazard.FIRE}
        cells = set(base_fire)
        for _ in range(phase % 4):
            cells = {(c, GRID - 1 - r) for r, c in cells}
        return cells

    def get_episode_stats(self):
        return {
            "turns_taken": self.turn_count,
            "deaths": self.death_count,
            "confused": self.confused_count,
            "cells_explored": len(set(self.cells_visited)),
            "path_length": len(self.cells_visited),
            "goal_reached": self.goal_reached,
        }

    def build_replay_payload(self, agent_overlay=None, dynamic_fire=True):
        positions = [list(self.start)] + [
            list(event["position"]) for event in self.replay_events
        ]
        deaths = [
            list(event["position"])
            for event in self.replay_events
            if event["died"]
        ]

        return {
            "version": 2,
            "episode": self.episode_number,
            "maze_name": self.maze_name,
            "maze_path": self.maze_path,
            "hazard_path": self.hazard_path,
            "start": list(self.start),
            "goal": list(self.goal),
            "positions": positions,
            "deaths": deaths,
            "dynamic_fire": dynamic_fire,
            "events": [
                {
                    **event,
                    "position": list(event["position"]),
                }
                for event in self.replay_events
            ],
            "summary": {
                **self.get_episode_stats(),
                "action_count": len(self.replay_events),
            },
            "agent_overlay": agent_overlay or {},
        }

    def export_replay(self, output_path, agent_overlay=None, dynamic_fire=True):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build_replay_payload(
            agent_overlay=agent_overlay,
            dynamic_fire=dynamic_fire,
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path
