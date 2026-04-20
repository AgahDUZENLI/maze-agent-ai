import argparse
import json
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from solver_environment import Action, MazeEnvironment
from maze_reader import DEFAULT_HAZARDS_PATH, DEFAULT_WALLS_PATH, GRID, load_maze

CELL_PX = 20
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SOLVER_DIR = RESULTS_DIR / "solver"
DEFAULT_KNOWLEDGE = RESULTS_DIR / "maze_knowledge.json"
KNOWLEDGE_FORMAT_VERSION = 1

DIRECTIONS = ("up", "right", "down", "left")
MOVE_DELTAS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
DIRECTION_TO_ACTION = {
    "up": Action.MOVE_UP,
    "down": Action.MOVE_DOWN,
    "left": Action.MOVE_LEFT,
    "right": Action.MOVE_RIGHT,
}
KNOWN_SAFE_TYPES = {"start", "safe", "goal", "confusion"}
NON_STANDABLE_TYPES = {"lethal", "teleport"}


def resolve_project_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def cell_key(cell):
    return f"{cell[0]},{cell[1]}"


def parse_cell_key(raw):
    text = str(raw).strip()
    if text.startswith("(") and text.endswith(")"):
        row_str, col_str = text[1:-1].split(",")
    else:
        row_str, col_str = text.split(",")
    return (int(row_str.strip()), int(col_str.strip()))


def serialize_cell_exits(cell_exits):
    payload = {}
    for cell, directions in sorted(cell_exits.items()):
        payload[cell_key(cell)] = {}
        for direction, stats in sorted(directions.items()):
            payload[cell_key(cell)][direction] = {
                "visits": int(stats.get("visits", 0)),
                "deaths": int(stats.get("deaths", 0)),
                "destination": list(stats["destination"]) if stats.get("destination") is not None else None,
            }
    return payload


def serialize_walls(walls):
    payload = {}
    for cell, directions in sorted(walls.items()):
        if directions:
            payload[cell_key(cell)] = sorted(directions)
    return payload


def serialize_cell_types(cell_types):
    return {cell_key(cell): value for cell, value in sorted(cell_types.items())}


def fresh_knowledge_state():
    return {
        "runs": 0,
        "move_history": [],
        "cell_exits": defaultdict(dict),
        "walls": defaultdict(set),
        "cell_types": {},
        "metadata": {},
    }


def load_knowledge(path=DEFAULT_KNOWLEDGE):
    if not path.exists():
        return fresh_knowledge_state()

    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return fresh_knowledge_state()

    metadata = dict(data.get("metadata", {}))
    if metadata.get("solver_state_version") != KNOWLEDGE_FORMAT_VERSION:
        return fresh_knowledge_state()

    cell_exits = defaultdict(dict)
    for raw_cell, directions in data.get("cell_exits", {}).items():
        cell = parse_cell_key(raw_cell)
        if isinstance(directions, list):
            directions = {
                direction: {"visits": 0, "deaths": 0, "destination": None}
                for direction in directions
            }
        for direction, stats in directions.items():
            destination = stats.get("destination")
            cell_exits[cell][direction] = {
                "visits": int(stats.get("visits", 0)),
                "deaths": int(stats.get("deaths", 0)),
                "destination": tuple(destination) if destination is not None else None,
            }

    walls = defaultdict(set)
    for raw_cell, directions in data.get("walls", {}).items():
        walls[parse_cell_key(raw_cell)].update(directions)

    cell_types = {}
    for raw_cell, value in data.get("cell_types", {}).items():
        cell_types[parse_cell_key(raw_cell)] = value

    return {
        "runs": int(data.get("runs", 0)),
        "move_history": list(data.get("move_history", [])),
        "cell_exits": cell_exits,
        "walls": walls,
        "cell_types": cell_types,
        "metadata": metadata,
    }


def save_knowledge(knowledge_path, knowledge, move_history, evaluation, metadata):
    knowledge_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "runs": len(move_history),
        "move_history": move_history,
        "cell_exits": serialize_cell_exits(knowledge["cell_exits"]),
        "walls": serialize_walls(knowledge["walls"]),
        "cell_types": serialize_cell_types(knowledge["cell_types"]),
        "evaluation": evaluation,
        "metadata": metadata,
    }

    with tempfile.NamedTemporaryFile(
        "w",
        dir=knowledge_path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2)
        temp_path = Path(handle.name)

    temp_path.replace(knowledge_path)


class Agent:
    def __init__(self, start, goal, knowledge):
        self.start = start
        self.goal = goal
        self.knowledge = knowledge
        self.knowledge["cell_types"].setdefault(start, "start")
        self.reset_episode()

    def reset_episode(self):
        self.current_position = self.start
        self.controls_inverted = False
        self.pending_turn = None
        self.move_log = []
        self.event_log = []
        self.wall_hits = 0
        self.deaths = 0
        self.teleports = 0
        self.confusion_entries = 0

    def _adjacent(self, cell, direction):
        dr, dc = MOVE_DELTAS[direction]
        return (cell[0] + dr, cell[1] + dc)

    def _ensure_cell(self, cell):
        self.knowledge["cell_exits"].setdefault(cell, {})
        self.knowledge["walls"].setdefault(cell, set())

    def _mark_cell_type(self, cell, cell_type):
        previous = self.knowledge["cell_types"].get(cell)
        if previous == "goal":
            return
        if previous == "start" and cell_type != "goal":
            return
        self.knowledge["cell_types"][cell] = cell_type

    def _mark_wall(self, cell, direction):
        self._ensure_cell(cell)
        self.knowledge["walls"][cell].add(direction)

    def _record_edge(self, origin, direction, destination, *, died=False):
        self._ensure_cell(origin)
        stats = self.knowledge["cell_exits"][origin].setdefault(
            direction,
            {"visits": 0, "deaths": 0, "destination": None},
        )
        stats["visits"] += 1
        stats["destination"] = destination
        if died:
            stats["deaths"] += 1

    def _record_reverse_edge_if_adjacent(self, origin, direction, destination):
        if destination != self._adjacent(origin, direction):
            return
        reverse = OPPOSITE[direction]
        self._ensure_cell(destination)
        reverse_stats = self.knowledge["cell_exits"][destination].setdefault(
            reverse,
            {"visits": 0, "deaths": 0, "destination": origin},
        )
        if reverse_stats.get("destination") is None:
            reverse_stats["destination"] = origin

    def _unknown_directions(self, cell):
        self._ensure_cell(cell)
        known_exits = set(self.knowledge["cell_exits"][cell])
        known_walls = set(self.knowledge["walls"][cell])
        return [direction for direction in DIRECTIONS if direction not in known_exits and direction not in known_walls]

    def _is_traversable(self, origin, direction, stats):
        destination = stats.get("destination")
        if destination is None:
            return False
        if stats.get("deaths", 0) > 0:
            return False
        cell_type = self.knowledge["cell_types"].get(destination)
        return cell_type != "lethal"

    def _route_to(self, targets):
        targets = set(targets)
        if not targets:
            return None
        if self.current_position in targets:
            return []

        queue = deque([self.current_position])
        previous = {self.current_position: None}
        action_taken = {}

        while queue:
            cell = queue.popleft()
            for direction, stats in self.knowledge["cell_exits"].get(cell, {}).items():
                if not self._is_traversable(cell, direction, stats):
                    continue
                destination = stats["destination"]
                if destination in previous:
                    continue
                previous[destination] = cell
                action_taken[destination] = direction
                if destination in targets:
                    route = []
                    cursor = destination
                    while previous[cursor] is not None:
                        route.append(action_taken[cursor])
                        cursor = previous[cursor]
                    route.reverse()
                    return route
                queue.append(destination)

        return None

    def _frontier_cells(self):
        frontier = set()
        for cell, cell_type in self.knowledge["cell_types"].items():
            if cell_type in NON_STANDABLE_TYPES:
                continue
            if self._unknown_directions(cell):
                frontier.add(cell)
        return frontier

    def _best_unknown_direction(self, cell):
        unknown = self._unknown_directions(cell)
        if not unknown:
            return None
        return min(
            unknown,
            key=lambda direction: (
                abs(self._adjacent(cell, direction)[0] - self.goal[0])
                + abs(self._adjacent(cell, direction)[1] - self.goal[1]),
                DIRECTIONS.index(direction),
            ),
        )

    def _encode_action(self, actual_direction):
        if self.controls_inverted:
            actual_direction = OPPOSITE[actual_direction]
        return DIRECTION_TO_ACTION[actual_direction]

    def _record_event(
        self,
        position,
        *,
        kind,
        consumes_move,
        teleported=False,
        confused=False,
        goal_reached=False,
    ):
        self.event_log.append(
            {
                "position": [position[0], position[1]],
                "kind": kind,
                "consumes_move": consumes_move,
                "teleported": teleported,
                "confused": confused,
                "goal_reached": goal_reached,
            }
        )

    def observe_turn_result(self, result):
        if self.pending_turn is None:
            self.current_position = self.start if result.is_dead else result.current_position
            self.controls_inverted = result.is_confused
            return

        origin = self.pending_turn["origin"]
        direction = self.pending_turn["direction"]
        attempted_cell = self._adjacent(origin, direction)
        self.wall_hits += result.wall_hits

        if result.wall_hits > 0:
            self._mark_wall(origin, direction)
            self.move_log.append([origin[0], origin[1]])
            self._record_event(origin, kind="wall", consumes_move=False)
            self.current_position = origin
            self.controls_inverted = False
            self.pending_turn = None
            return

        if result.is_dead:
            self._record_edge(origin, direction, attempted_cell, died=True)
            self._mark_cell_type(attempted_cell, "lethal")
            self.deaths += 1
            self.move_log.append([attempted_cell[0], attempted_cell[1]])
            self._record_event(attempted_cell, kind="death", consumes_move=True)
            self.move_log.append([self.start[0], self.start[1]])
            self._record_event(self.start, kind="respawn", consumes_move=False)
            self.current_position = self.start
            self.controls_inverted = False
            self.pending_turn = None
            return

        self._record_edge(origin, direction, result.current_position)

        if result.teleported:
            self.teleports += 1
            self._mark_cell_type(attempted_cell, "teleport")
        else:
            self._record_reverse_edge_if_adjacent(origin, direction, result.current_position)

        if result.is_goal_reached:
            self._mark_cell_type(result.current_position, "goal")
        elif result.is_confused:
            self.confusion_entries += 1
            self._mark_cell_type(result.current_position, "confusion")
        else:
            self._mark_cell_type(result.current_position, "safe")

        self.move_log.append([result.current_position[0], result.current_position[1]])
        self._record_event(
            result.current_position,
            kind="teleport" if result.teleported else "move",
            consumes_move=True,
            teleported=result.teleported,
            confused=result.is_confused,
            goal_reached=result.is_goal_reached,
        )
        self.current_position = result.current_position
        self.controls_inverted = result.is_confused
        self.pending_turn = None

    def plan_turn(self, last_result):
        if last_result is not None:
            self.observe_turn_result(last_result)

        if self.current_position == self.goal:
            return []

        route_to_goal = self._route_to({self.goal})
        if route_to_goal:
            chosen_direction = route_to_goal[0]
        else:
            chosen_direction = self._best_unknown_direction(self.current_position)
            if chosen_direction is None:
                route_to_frontier = self._route_to(self._frontier_cells())
                if route_to_frontier:
                    chosen_direction = route_to_frontier[0]

        if chosen_direction is None:
            return []

        issued_action = self._encode_action(chosen_direction)
        self.pending_turn = {
            "origin": self.current_position,
            "direction": chosen_direction,
            "issued_action": issued_action,
        }
        return [issued_action]

    def build_run_record(self, run_number, runtime_seconds, env_stats, success):
        exploration_efficiency = 0.0
        if env_stats["total_cells_visited"] > 0:
            exploration_efficiency = env_stats["cells_explored"] / env_stats["total_cells_visited"]

        return {
            "run": run_number,
            "success": success,
            "turns": env_stats["turns_taken"],
            "steps": env_stats["path_length"],
            "wall_hits": self.wall_hits,
            "deaths": env_stats["deaths"],
            "teleports": self.teleports,
            "confusion_entries": self.confusion_entries,
            "cells_explored": env_stats["cells_explored"],
            "total_cells_visited": env_stats["total_cells_visited"],
            "exploration_efficiency": exploration_efficiency,
            "runtime_seconds": runtime_seconds,
            "moves": self.move_log,
            "events": self.event_log,
        }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def evaluate_agent(environment, agent, episodes, max_turns, starting_run):
    run_records = []

    for episode_index in range(episodes):
        environment.reset()
        agent.reset_episode()
        pending_result = None
        success = False
        started_at = time.time()

        for _ in range(max_turns):
            actions = agent.plan_turn(pending_result)
            pending_result = None

            if not actions:
                break

            result = environment.step(actions)
            if result.is_goal_reached:
                agent.observe_turn_result(result)
                success = True
                break

            pending_result = result

        if pending_result is not None:
            agent.observe_turn_result(pending_result)

        env_stats = environment.get_episode_stats()
        run_number = starting_run + episode_index + 1
        record = agent.build_run_record(
            run_number=run_number,
            runtime_seconds=time.time() - started_at,
            env_stats=env_stats,
            success=success,
        )
        run_records.append(record)

        print(
            f"Episode {run_number}: "
            f"{'SUCCESS' if success else 'FAILED'} | "
            f"turns={record['turns']} steps={record['steps']} "
            f"deaths={record['deaths']} walls={record['wall_hits']} "
            f"explore={record['exploration_efficiency']:.3f}"
        )

    evaluation = {
        "episodes": len(run_records),
        "success_rate": mean([1.0 if record["success"] else 0.0 for record in run_records]),
        "avg_turns": mean([record["turns"] for record in run_records]),
        "avg_path_length": mean([record["steps"] for record in run_records]),
        "avg_deaths": mean([record["deaths"] for record in run_records]),
        "avg_wall_hits": mean([record["wall_hits"] for record in run_records]),
        "avg_exploration_efficiency": mean(
            [record["exploration_efficiency"] for record in run_records]
        ),
    }

    return run_records, evaluation


def reconstruct_shortest_known_path(cell_exits, start, goal):
    queue = deque([start])
    previous = {start: None}
    action_taken = {}

    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        for direction, stats in cell_exits.get(cell, {}).items():
            destination = stats.get("destination")
            if destination is None or stats.get("deaths", 0) > 0:
                continue
            if destination in previous:
                continue
            previous[destination] = cell
            action_taken[destination] = direction
            queue.append(destination)

    if goal not in previous:
        return None

    path = [goal]
    cursor = goal
    while previous[cursor] is not None:
        cursor = previous[cursor]
        path.append(cursor)
    path.reverse()
    return path


def render_path(h_walls, v_walls, path, start, goal, run_number, *, optimized):
    output_path = SOLVER_DIR / f"path_run{run_number}_{'optimized' if optimized else 'blind'}.png"
    size = GRID * CELL_PX
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 10)
    except Exception:
        font = ImageFont.load_default()

    for row in range(GRID):
        for col in range(GRID):
            x0, y0 = col * CELL_PX, row * CELL_PX
            if (row, col) == start:
                fill = (0, 200, 0)
            elif (row, col) == goal:
                fill = (200, 0, 0)
            else:
                fill = (240, 240, 240)
            draw.rectangle([x0, y0, x0 + CELL_PX, y0 + CELL_PX], fill=fill)

    for wall_index in range(GRID + 1):
        for col in range(GRID):
            if h_walls[wall_index, col]:
                draw.rectangle(
                    [col * CELL_PX, wall_index * CELL_PX, (col + 1) * CELL_PX, wall_index * CELL_PX + 2],
                    fill=(0, 0, 0),
                )

    for row in range(GRID):
        for wall_index in range(GRID + 1):
            if v_walls[row, wall_index]:
                draw.rectangle(
                    [wall_index * CELL_PX, row * CELL_PX, wall_index * CELL_PX + 2, (row + 1) * CELL_PX],
                    fill=(0, 0, 0),
                )

    def center(cell):
        row, col = cell
        return (col * CELL_PX + CELL_PX // 2, row * CELL_PX + CELL_PX // 2)

    color = (0, 100, 255) if optimized else (255, 0, 0)
    width = 4 if optimized else 2

    for index in range(len(path) - 1):
        draw.line([center(path[index]), center(path[index + 1])], fill=color, width=width)

    for index, (row, col) in enumerate(path):
        if optimized and index not in {0, len(path) - 1}:
            continue
        draw.text((col * CELL_PX + 2, row * CELL_PX + 2), str(index), fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the spec-aligned maze solver.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes to run.")
    parser.add_argument("--max-turns", type=int, default=20000, help="Hard turn cap per episode.")
    parser.add_argument("--walls", default=DEFAULT_WALLS_PATH, help="Wall image path, relative to the project root by default.")
    parser.add_argument("--hazards", default=DEFAULT_HAZARDS_PATH, help="Hazard image path, relative to the project root by default.")
    parser.add_argument("--teleports", default=None, help="Optional teleport mapping JSON for non-paired deterministic teleport layouts.")
    parser.add_argument("--knowledge", default=str(DEFAULT_KNOWLEDGE.relative_to(BASE_DIR)), help="Knowledge JSON path, relative to the project root by default.")
    parser.add_argument("--reset-knowledge", action="store_true", help="Ignore any existing knowledge file and start this maze from scratch.")
    parser.add_argument("--no-render", action="store_true", help="Skip saving path render images.")
    return parser.parse_args()


def main():
    args = parse_args()
    walls_path = resolve_project_path(args.walls)
    hazards_path = resolve_project_path(args.hazards)
    knowledge_path = resolve_project_path(args.knowledge)
    teleport_path = resolve_project_path(args.teleports) if args.teleports else None

    knowledge = fresh_knowledge_state() if args.reset_knowledge else load_knowledge(knowledge_path)
    current_teleport_model = "explicit-map" if teleport_path else "asset-paired-fallback"
    prior_metadata = knowledge.get("metadata", {})
    if (
        prior_metadata.get("walls_path")
        and (
            prior_metadata.get("walls_path") != str(walls_path)
            or prior_metadata.get("hazards_path") != str(hazards_path)
            or prior_metadata.get("teleport_model") != current_teleport_model
        )
    ):
        knowledge = fresh_knowledge_state()

    environment = MazeEnvironment(
        maze_id=walls_path.stem,
        walls_path=walls_path,
        hazards_path=hazards_path,
        teleport_map_path=teleport_path,
    )
    agent = Agent(environment.start, environment.goal, knowledge)

    run_records, evaluation = evaluate_agent(
        environment=environment,
        agent=agent,
        episodes=args.episodes,
        max_turns=args.max_turns,
        starting_run=knowledge["runs"],
    )

    move_history = knowledge["move_history"] + run_records
    metadata = {
        "solver_state_version": KNOWLEDGE_FORMAT_VERSION,
        "solver_mode": "spec-partial-observation",
        "hazard_model": "static",
        "teleport_model": current_teleport_model,
        "walls_path": str(walls_path),
        "hazards_path": str(hazards_path),
    }
    save_knowledge(knowledge_path, knowledge, move_history, evaluation, metadata)

    print("\nAggregate:")
    print(f"  episodes               : {evaluation['episodes']}")
    print(f"  success_rate           : {evaluation['success_rate']:.3f}")
    print(f"  avg_turns              : {evaluation['avg_turns']:.2f}")
    print(f"  avg_path_length        : {evaluation['avg_path_length']:.2f}")
    print(f"  avg_deaths             : {evaluation['avg_deaths']:.2f}")
    print(f"  avg_wall_hits          : {evaluation['avg_wall_hits']:.2f}")
    print(f"  avg_exploration_eff    : {evaluation['avg_exploration_efficiency']:.3f}")

    if args.no_render or not run_records:
        return

    _, h_walls, v_walls = load_maze(str(walls_path))
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


if __name__ == "__main__":
    main()
