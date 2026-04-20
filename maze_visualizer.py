import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pygame
from PIL import Image
from maze_reader import (
    DEFAULT_HAZARDS_PATH,
    DEFAULT_WALLS_PATH,
    GRID,
    Hazard,
    can_move,
    get_goal,
    get_start,
    load_hazards,
    load_maze,
    update_fire_in_hazards,
)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
VISUALIZER_DIR = RESULTS_DIR / "visualizer"
DEFAULT_KNOWLEDGE = RESULTS_DIR / "maze_knowledge.json"

BG_COL = (12, 14, 19)
BOARD_BG = (20, 24, 31)
BOARD_TILE = (30, 35, 44)
HUD_BG = (245, 239, 228)
HUD_BORDER = (220, 212, 198)
TEXT_DARK = (24, 24, 24)
TEXT_MUTED = (106, 102, 94)
WALL_COL = (228, 230, 234)
START_COL = (62, 182, 108)
GOAL_COL = (228, 92, 92)
PATH_FAINT = (255, 155, 155)
PATH_RECENT = (114, 182, 255)
AGENT_COL = (255, 255, 255)
AGENT_CORE = (58, 124, 255)
WARNING_COL = (255, 166, 45)
COLLISION_COL = (235, 55, 55)
PROGRESS_FILL = (63, 145, 112)
TELEPORT_COL = (160, 104, 255)
CONFUSION_COL = (61, 171, 214)

HAZARD_FILL = {
    Hazard.FIRE: (255, 135, 75),
    Hazard.CONFUSION: (90, 205, 255),
    Hazard.TP_GREEN: (70, 215, 125),
    Hazard.TP_YELLOW: (255, 214, 72),
    Hazard.TP_PURPLE: (187, 121, 255),
    Hazard.TP_RED: (255, 115, 115),
}

HAZARD_LABELS = [
    (Hazard.FIRE, "Fire"),
    (Hazard.CONFUSION, "Confusion"),
    (Hazard.TP_GREEN, "Green TP"),
    (Hazard.TP_YELLOW, "Yellow TP"),
    (Hazard.TP_PURPLE, "Purple TP"),
    (Hazard.TP_RED, "Red TP"),
]

DIRECTION_DELTAS = {
    (-1, 0): "up",
    (0, 1): "right",
    (1, 0): "down",
    (0, -1): "left",
}


@dataclass
class FrameState:
    index: int
    cell: tuple
    path_prefix: list
    move_count: int
    turn_count: int
    move_in_turn: int
    hazards: dict
    fire_distance: Optional[int]
    on_fire_now: bool
    fatal_fire_hit: bool
    fire_update_after_move: bool
    entered_hazard: Optional[str]
    transition_kind: str
    respawned: bool
    reached_goal: bool
    event_tags: list


def resolve_project_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def load_run_entry(knowledge_path, run_number=None):
    if not knowledge_path.exists():
        raise FileNotFoundError(f"Knowledge file not found: {knowledge_path}")

    with open(knowledge_path) as f:
        data = json.load(f)

    move_history = data.get("move_history", [])
    if not move_history:
        raise ValueError(f"No move history found in {knowledge_path}")

    if run_number is None:
        return move_history[-1]

    for entry in move_history:
        if entry.get("run") == run_number:
            return entry

    raise ValueError(f"Run {run_number} not found in {knowledge_path}")


def load_knowledge_payload(knowledge_path):
    if not knowledge_path.exists():
        raise FileNotFoundError(f"Knowledge file not found: {knowledge_path}")
    with open(knowledge_path) as f:
        return json.load(f)


def direction_name(prev_cell, next_cell):
    dr = next_cell[0] - prev_cell[0]
    dc = next_cell[1] - prev_cell[1]
    return DIRECTION_DELTAS.get((dr, dc))


def transition_is_move(prev_cell, next_cell, h_walls, v_walls):
    direction = direction_name(prev_cell, next_cell)
    if direction is None:
        return False
    return can_move(prev_cell[0], prev_cell[1], direction, h_walls, v_walls)


def fire_proximity(cell, hazards):
    fire_cells = [hazard_cell for hazard_cell, hazard in hazards.items() if hazard == Hazard.FIRE]
    if not fire_cells:
        return None, False
    if hazards.get(cell) == Hazard.FIRE:
        return 0, True
    min_distance = min(abs(cell[0] - row) + abs(cell[1] - col) for row, col in fire_cells)
    return min_distance, False


def choose_stride(total_positions, max_frames):
    if total_positions <= max_frames:
        return 1
    return max(1, math.ceil(total_positions / max_frames))


def choose_dynamic_stride(replay, max_frames):
    total_positions = len(replay["frames"])
    if replay["summary"]["fatal_fire_hits"] == 0 and replay["summary"]["logged_positions"] <= 500:
        return 1
    return choose_stride(total_positions, max_frames)


def classify_transition(prev_cell, current_cell, h_walls, v_walls):
    if prev_cell == current_cell:
        return "pause"
    if transition_is_move(prev_cell, current_cell, h_walls, v_walls):
        return "move"
    return "jump"


def hazard_name(hazard):
    if hazard is None:
        return None
    labels = {
        Hazard.FIRE: "fire",
        Hazard.CONFUSION: "confusion",
        Hazard.TP_GREEN: "green teleport",
        Hazard.TP_YELLOW: "yellow teleport",
        Hazard.TP_PURPLE: "purple teleport",
        Hazard.TP_RED: "red teleport",
    }
    return labels.get(hazard)


def load_event_log(run_entry, logged_path):
    events = run_entry.get("events")
    if not isinstance(events, list) or len(events) != len(logged_path):
        return None
    return events


def event_consumes_move(event, prev_cell, next_cell, h_walls, v_walls):
    if event is not None:
        return bool(event.get("consumes_move", False))

    if prev_cell == next_cell:
        return False
    if transition_is_move(prev_cell, next_cell, h_walls, v_walls):
        return True
    return False


def transition_consumes_move(event, prev_cell, next_cell, h_walls, v_walls, entered_hazard):
    if event is not None:
        return event_consumes_move(event, prev_cell, next_cell, h_walls, v_walls)

    if event_consumes_move(None, prev_cell, next_cell, h_walls, v_walls):
        return True

    return entered_hazard in {
        "teleport",
        "green teleport",
        "yellow teleport",
        "purple teleport",
        "red teleport",
    }


def event_transition_kind(event, prev_cell, next_cell, h_walls, v_walls):
    if event is not None:
        kind = event.get("kind")
        if kind == "wall":
            return "pause"
        if kind in {"teleport", "respawn"}:
            return "jump"
        if kind in {"move", "death"}:
            return "move" if transition_is_move(prev_cell, next_cell, h_walls, v_walls) else "jump"

    return classify_transition(prev_cell, next_cell, h_walls, v_walls)


def event_entered_hazard(event, next_cell, hazards_now):
    if event is not None:
        if event.get("confused"):
            return "confusion"
        if event.get("teleported"):
            hazard_label = hazard_name(hazards_now.get(next_cell))
            return hazard_label if hazard_label is not None else "teleport"
        if event.get("kind") == "death":
            return "fire"
    return hazard_name(hazards_now.get(next_cell))


def event_fatal_fire_hit(event, next_cell, hazards_now):
    if event is not None:
        return event.get("kind") == "death"
    return hazards_now.get(next_cell) == Hazard.FIRE


def event_respawned(event, prev_cell, next_cell, start):
    if event is not None:
        return event.get("kind") == "respawn"
    return prev_cell != start and next_cell == start


def build_event_tags(frame):
    tags = []
    if frame.index == 0:
        tags.append("start")
    if frame.transition_kind == "jump" and frame.entered_hazard and "teleport" in frame.entered_hazard:
        tags.append("teleport")
    if frame.respawned:
        tags.append("respawn")
    if frame.entered_hazard == "confusion":
        tags.append("confusion")
    if frame.fire_update_after_move:
        tags.append("fire update")
    if frame.fatal_fire_hit:
        tags.append("fatal fire")
    elif frame.on_fire_now:
        tags.append("on fire now")
    elif frame.fire_distance == 1:
        tags.append("near fire")
    if frame.entered_hazard and frame.entered_hazard not in {"confusion"} and "teleport" not in frame.entered_hazard:
        tags.append(frame.entered_hazard)
    return tags


def format_status(frame):
    if frame.reached_goal:
        return "Goal reached", PROGRESS_FILL
    if frame.respawned:
        return "Respawned after death", COLLISION_COL
    if frame.fatal_fire_hit:
        return "Fatal fire landing", COLLISION_COL
    if frame.on_fire_now:
        return "On fire after turn update", COLLISION_COL
    if frame.fire_distance == 1:
        return "Near-fire pass", WARNING_COL
    if frame.entered_hazard == "confusion":
        return "Confusion tile entered", CONFUSION_COL
    if frame.entered_hazard and "teleport" in frame.entered_hazard:
        return f"Used {frame.entered_hazard}", TELEPORT_COL
    if frame.fire_update_after_move:
        return "Fire rotated this turn", WARNING_COL
    return "Stable route", PROGRESS_FILL


def build_replay(walls_path, hazards_path, knowledge_path, run_number=None, force_dynamic_fire=False):
    _, h_walls, v_walls = load_maze(str(walls_path))
    hazards = load_hazards(str(hazards_path))
    start = get_start(h_walls)
    goal = get_goal(h_walls)

    knowledge_payload = load_knowledge_payload(knowledge_path)
    run_entry = load_run_entry(knowledge_path, run_number)
    logged_path = [tuple(cell) for cell in run_entry["moves"]]
    event_log = load_event_log(run_entry, logged_path)
    full_path = [start] + logged_path
    run_succeeded = bool(full_path and full_path[-1] == goal)
    actual_hazard_model = knowledge_payload.get("metadata", {}).get("hazard_model", "dynamic")
    display_hazard_model = "dynamic" if force_dynamic_fire else actual_hazard_model

    display_frame_hazards = [dict(hazards)]
    actual_frame_hazards = [dict(hazards)]
    fatal_fire_by_index = [False]
    entered_hazard_by_index = [None]
    fire_update_after_index = [False]
    transition_kind_by_index = ["start"]
    actual_hazards = dict(hazards)
    display_hazards = dict(hazards)
    actual_fire_pivots = None
    display_fire_pivots = None
    move_count = 0

    for index in range(len(full_path) - 1):
        cell = full_path[index]
        next_cell = full_path[index + 1]
        event = event_log[index] if event_log is not None else None
        entered_hazard = event_entered_hazard(event, next_cell, actual_hazards)
        consumes_move = transition_consumes_move(
            event,
            cell,
            next_cell,
            h_walls,
            v_walls,
            entered_hazard,
        )
        transition_kind = event_transition_kind(event, cell, next_cell, h_walls, v_walls)
        fatal_fire_hit = event_fatal_fire_hit(event, next_cell, actual_hazards)

        next_actual_hazards = actual_hazards
        next_display_hazards = display_hazards
        actual_fire_update = False

        if consumes_move:
            move_count += 1
            if actual_hazard_model == "dynamic" and move_count % 5 == 0:
                next_actual_hazards, actual_fire_pivots = update_fire_in_hazards(
                    actual_hazards, actual_fire_pivots
                )
                actual_fire_update = True
            if display_hazard_model == "dynamic" and move_count % 5 == 0:
                next_display_hazards, display_fire_pivots = update_fire_in_hazards(
                    display_hazards, display_fire_pivots
                )
        actual_frame_hazards.append(dict(next_actual_hazards))
        display_frame_hazards.append(dict(next_display_hazards))

        fatal_fire_by_index.append(fatal_fire_hit)
        entered_hazard_by_index.append(entered_hazard)
        fire_update_after_index.append(actual_fire_update)
        transition_kind_by_index.append(transition_kind)
        actual_hazards = next_actual_hazards
        display_hazards = next_display_hazards

    frames = []
    move_count = 0
    for index, cell in enumerate(full_path):
        display_hazards_now = display_frame_hazards[index]
        actual_hazards_now = actual_frame_hazards[index]
        fatal_fire_hit = fatal_fire_by_index[index]
        fire_distance, on_fire_now = fire_proximity(cell, actual_hazards_now)
        event = event_log[index - 1] if event_log is not None and index > 0 else None
        if index > 0 and transition_consumes_move(
            event,
            full_path[index - 1],
            cell,
            h_walls,
            v_walls,
            entered_hazard_by_index[index],
        ):
            move_count += 1
        move_in_turn = ((move_count - 1) % 5) + 1 if move_count else 0
        respawned = (
            event_respawned(event, full_path[index - 1], cell, start)
            if index > 0
            else False
        )
        reached_goal = bool(event.get("goal_reached")) if event is not None else cell == goal

        frame = FrameState(
            index=index,
            cell=cell,
            path_prefix=full_path[: index + 1],
            move_count=move_count,
            turn_count=math.ceil(move_count / 5) if move_count else 0,
            move_in_turn=move_in_turn,
            hazards=display_hazards_now,
            fire_distance=fire_distance,
            on_fire_now=on_fire_now,
            fatal_fire_hit=fatal_fire_hit,
            fire_update_after_move=fire_update_after_index[index],
            entered_hazard=entered_hazard_by_index[index],
            transition_kind=transition_kind_by_index[index],
            respawned=respawned,
            reached_goal=reached_goal,
            event_tags=[],
        )
        frame.event_tags = build_event_tags(frame)
        frames.append(frame)

    return {
        "run_number": run_entry["run"],
        "start": start,
        "goal": goal,
        "h_walls": h_walls,
        "v_walls": v_walls,
        "frames": frames,
        "summary": {
            "succeeded": run_succeeded,
            "hazard_model": actual_hazard_model,
            "display_hazard_model": display_hazard_model,
            "fatal_fire_hits": sum(1 for frame in frames if frame.fatal_fire_hit),
            "on_fire_now": sum(1 for frame in frames if frame.on_fire_now),
            "teleports": sum(
                1
                for frame in frames
                if frame.transition_kind == "jump"
                and frame.entered_hazard in {"teleport", "green teleport", "yellow teleport", "purple teleport", "red teleport"}
            ),
            "confusion_entries": sum(1 for frame in frames if frame.entered_hazard == "confusion"),
            "fire_updates": sum(1 for frame in frames if frame.fire_update_after_move),
            "respawns": sum(1 for frame in frames if frame.respawned),
            "min_fire_distance": min(
                [frame.fire_distance for frame in frames if frame.fire_distance is not None],
                default=None,
            ),
            "logged_positions": len(logged_path),
        },
    }


def surface_to_image(surface):
    data = pygame.image.tostring(surface, "RGB")
    return Image.frombytes("RGB", surface.get_size(), data)


def draw_rounded_panel(surface, rect, color, border):
    pygame.draw.rect(surface, color, rect, border_radius=18)
    pygame.draw.rect(surface, border, rect, width=2, border_radius=18)


def fit_text(text, font, max_width):
    if font.size(text)[0] <= max_width:
        return text

    clipped = text
    while clipped and font.size(f"{clipped}...")[0] > max_width:
        clipped = clipped[:-1]
    return f"{clipped}..." if clipped else "..."


def draw_hazard_legend(surface, frame, x, y, width, fonts):
    label_font, value_font = fonts
    counts = {hazard: 0 for hazard, _ in HAZARD_LABELS}
    for hazard in frame.hazards.values():
        counts[hazard] += 1

    surface.blit(label_font.render("Hazards This Frame", True, TEXT_DARK), (x, y))
    row_y = y + 30
    value_x = x + width
    label_width = max(72, width - 52)
    for hazard, label in HAZARD_LABELS:
        pygame.draw.rect(surface, HAZARD_FILL[hazard], pygame.Rect(x, row_y + 3, 18, 18), border_radius=4)
        clipped_label = fit_text(label, value_font, label_width)
        surface.blit(value_font.render(clipped_label, True, TEXT_MUTED), (x + 28, row_y))
        value = value_font.render(str(counts[hazard]), True, TEXT_DARK)
        surface.blit(value, (value_x - value.get_width(), row_y))
        row_y += 24


def draw_event_log(surface, frame, x, y, width, fonts):
    title_font, body_font = fonts
    surface.blit(title_font.render("Events", True, TEXT_DARK), (x, y))
    events = []
    if frame.index == 0:
        events.append(("Replay start", TEXT_MUTED))
    if frame.reached_goal:
        events.append(("Goal reached", PROGRESS_FILL))
    if frame.respawned:
        events.append(("Respawned at start after death", COLLISION_COL))
    if frame.transition_kind == "jump" and frame.entered_hazard and "teleport" in frame.entered_hazard:
        events.append((f"Teleport event: {frame.entered_hazard}", TELEPORT_COL))
    if frame.entered_hazard == "confusion":
        events.append(("Entered confusion tile", CONFUSION_COL))
    if frame.fire_update_after_move:
        events.append((f"Fire rotated after move {frame.move_count}", WARNING_COL))
    if frame.fatal_fire_hit:
        events.append(("Fatal fire landing", COLLISION_COL))
    elif frame.on_fire_now:
        events.append(("Standing on fire after update", COLLISION_COL))
    elif frame.fire_distance == 1:
        events.append(("One cell from fire", WARNING_COL))
    if frame.index == 0:
        pass
    elif not events:
        events.append(("No special event on this frame", TEXT_MUTED))

    row_y = y + 30
    bullet_width = body_font.size("• ")[0]
    max_text_width = max(40, width - bullet_width)
    for text, color in events[:3]:
        wrapped = fit_text(text, body_font, max_text_width)
        surface.blit(body_font.render(f"• {wrapped}", True, color), (x, row_y))
        row_y += 22


def draw_run_summary(surface, summary, x, y, width, fonts):
    title_font, body_font = fonts
    surface.blit(title_font.render("Run Summary", True, TEXT_DARK), (x, y))

    summary_lines = [
        f"Outcome: {'success' if summary['succeeded'] else 'failed'}",
        f"Fatal / On fire: {summary['fatal_fire_hits']} / {summary['on_fire_now']}",
        f"Respawn / TP: {summary['respawns']} / {summary['teleports']}",
        f"Confuse / Updates: {summary['confusion_entries']} / {summary['fire_updates']}",
        f"Min fire / Logged: {summary['min_fire_distance']} / {summary['logged_positions']}",
    ]

    row_y = y + 30
    for line in summary_lines:
        clipped = fit_text(line, body_font, width)
        surface.blit(body_font.render(clipped, True, TEXT_MUTED), (x, row_y))
        row_y += 20


def draw_board(surface, replay, frame, board_x, board_y, cell_px):
    board_rect = pygame.Rect(board_x, board_y, GRID * cell_px, GRID * cell_px)
    pygame.draw.rect(surface, BOARD_BG, board_rect, border_radius=18)

    for row in range(GRID):
        for col in range(GRID):
            rect = pygame.Rect(board_x + col * cell_px, board_y + row * cell_px, cell_px, cell_px)
            fill = BOARD_TILE
            if (row, col) == replay["start"]:
                fill = START_COL
            elif (row, col) == replay["goal"]:
                fill = GOAL_COL
            pygame.draw.rect(surface, fill, rect)

    for cell, hazard in frame.hazards.items():
        row, col = cell
        rect = pygame.Rect(board_x + col * cell_px + 1, board_y + row * cell_px + 1, cell_px - 2, cell_px - 2)
        pygame.draw.rect(surface, HAZARD_FILL[hazard], rect, border_radius=3)

    for wi in range(GRID + 1):
        for col in range(GRID):
            if replay["h_walls"][wi, col]:
                x = board_x + col * cell_px
                y = board_y + wi * cell_px
                pygame.draw.line(surface, WALL_COL, (x, y), (x + cell_px, y), max(2, cell_px // 4))
    for row in range(GRID):
        for wi in range(GRID + 1):
            if replay["v_walls"][row, wi]:
                x = board_x + wi * cell_px
                y = board_y + row * cell_px
                pygame.draw.line(surface, WALL_COL, (x, y), (x, y + cell_px), max(2, cell_px // 4))

    centers = [
        (
            board_x + cell[1] * cell_px + cell_px // 2,
            board_y + cell[0] * cell_px + cell_px // 2,
        )
        for cell in frame.path_prefix
    ]
    if len(centers) > 1:
        pygame.draw.lines(surface, PATH_FAINT, False, centers, max(2, cell_px // 4))
        recent = centers[max(0, len(centers) - 36):]
        if len(recent) > 1:
            pygame.draw.lines(surface, PATH_RECENT, False, recent, max(3, cell_px // 3))

    agent_x, agent_y = centers[-1]
    pulse = 2 + (frame.index % 6)
    if frame.fatal_fire_hit:
        pygame.draw.circle(surface, COLLISION_COL, (agent_x, agent_y), cell_px // 2 + 9 + pulse // 2, 5)
    elif frame.on_fire_now:
        pygame.draw.circle(surface, COLLISION_COL, (agent_x, agent_y), cell_px // 2 + 8, 4)
    elif frame.fire_distance == 1:
        pygame.draw.circle(surface, WARNING_COL, (agent_x, agent_y), cell_px // 2 + 7, 4)

    pygame.draw.circle(surface, AGENT_COL, (agent_x, agent_y), max(6, cell_px // 2))
    pygame.draw.circle(surface, AGENT_CORE, (agent_x, agent_y), max(3, cell_px // 3))


def draw_hud(surface, replay, frame, stride, hud_rect, fonts):
    title_font, section_font, body_font = fonts
    draw_rounded_panel(surface, hud_rect, HUD_BG, HUD_BORDER)
    status, status_col = format_status(frame)

    x = hud_rect.x + 24
    y = hud_rect.y + 24
    surface.blit(title_font.render("Maze Replay", True, TEXT_DARK), (x, y))
    surface.blit(body_font.render(status, True, status_col), (x, y + 34))
    outcome_text = "SUCCESSFUL RUN" if replay["summary"]["succeeded"] else "FAILED RUN"
    outcome_col = PROGRESS_FILL if replay["summary"]["succeeded"] else COLLISION_COL
    surface.blit(body_font.render(outcome_text, True, outcome_col), (x, y + 58))

    info_lines = [
        f"Run {replay['run_number']}",
        f"Step {frame.index}/{replay['frames'][-1].index}",
        f"Moves {frame.move_count}",
        f"Turn {frame.turn_count}",
        f"Move in turn {frame.move_in_turn}/5" if frame.move_count else "Move in turn 0/5",
        f"Nearest fire {frame.fire_distance if frame.fire_distance is not None else 'n/a'}",
        f"Hazards {replay['summary']['hazard_model']} -> {replay['summary']['display_hazard_model']}",
        f"Sample stride {stride}",
    ]
    row_y = y + 106
    for line in info_lines:
        surface.blit(body_font.render(line, True, TEXT_MUTED), (x, row_y))
        row_y += 21

    content_width = hud_rect.width - 48
    events_y = row_y + 6
    draw_event_log(surface, frame, x, events_y, content_width, (section_font, body_font))

    hazards_y = events_y + 100
    draw_hazard_legend(surface, frame, x, hazards_y, content_width, (section_font, body_font))

    summary_y = hazards_y + 174
    draw_run_summary(surface, replay["summary"], x, summary_y, content_width, (section_font, body_font))


def draw(surface, replay, frame, stride, cell_px, board_x, board_y, hud_rect, fonts):
    surface.fill(BG_COL)
    draw_board(surface, replay, frame, board_x, board_y, cell_px)
    draw_hud(surface, replay, frame, stride, hud_rect, fonts)


def frame_duration_for(frame, default_duration):
    duration = max(default_duration, 85)
    if frame.index == 0:
        return max(duration * 4, 400)
    if frame.fatal_fire_hit or frame.on_fire_now:
        return max(duration * 4, 420)
    if frame.transition_kind == "jump":
        return max(duration * 3, 320)
    if frame.entered_hazard == "confusion":
        return max(duration * 3, 280)
    if frame.fire_update_after_move:
        return max(duration * 3, 260)
    if frame.fire_distance == 1:
        return max(duration * 2, 180)
    return duration


def render_replay(replay, output_path, max_frames=220, cell_px=12, frame_duration_ms=90):
    pygame.init()
    try:
        frames = replay["frames"]
        stride = choose_dynamic_stride(replay, max_frames=max_frames)
        selected_frames = [frames[i] for i in range(0, len(frames), stride)]
        if selected_frames[-1].index != frames[-1].index:
            selected_frames.append(frames[-1])

        board_x = 28
        board_y = 28
        board_size = GRID * cell_px
        hud_rect = pygame.Rect(board_x + board_size + 24, 28, 310, board_size)
        width = hud_rect.right + 28
        height = board_y + board_size + 28

        surface = pygame.Surface((width, height))
        fonts = (
            pygame.font.SysFont("Avenir Next", 28, bold=True),
            pygame.font.SysFont("Avenir Next", 20, bold=True),
            pygame.font.SysFont("Avenir Next", 18),
        )

        pil_frames = []
        durations = []
        for frame in selected_frames:
            draw(surface, replay, frame, stride, cell_px, board_x, board_y, hud_rect, fonts)
            pil_frames.append(surface_to_image(surface))
            durations.append(frame_duration_for(frame, frame_duration_ms))

        if durations:
            durations[-1] = max(durations[-1], 1100)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations,
            loop=0,
            optimize=False,
        )
        return {
            "frame_count": len(selected_frames),
            "sample_stride": stride,
        }
    finally:
        pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(description="Render a pygame replay from the latest maze solver results.")
    parser.add_argument("--run", type=int, default=None, help="Run number from results/maze_knowledge.json. Defaults to latest.")
    parser.add_argument("--walls", default=DEFAULT_WALLS_PATH, help="Wall image path, relative to the project root by default.")
    parser.add_argument("--hazards", default=DEFAULT_HAZARDS_PATH, help="Hazard image path, relative to the project root by default.")
    parser.add_argument("--knowledge", default=str(DEFAULT_KNOWLEDGE.relative_to(BASE_DIR)), help="Knowledge JSON path, relative to the project root by default.")
    parser.add_argument("--output", default=None, help="Output GIF path. Defaults to results/visualizer/pygame_replay_runN.gif.")
    parser.add_argument("--max-frames", type=int, default=220, help="Maximum number of frames before sampling is applied.")
    parser.add_argument("--cell-px", type=int, default=12, help="Cell size in pixels for the board.")
    parser.add_argument("--duration", type=int, default=90, help="Base frame duration in milliseconds.")
    parser.add_argument("--rotate-fire", action="store_true", help="Compatibility flag. Rotating fire is already enabled by default.")
    parser.add_argument("--static-fire", action="store_true", help="Render the saved hazards without applying display-only fire rotation.")
    parser.add_argument("--no-latest-alias", action="store_true", help="Skip writing results/pygame_replay_latest.gif.")
    return parser.parse_args()


def main():
    args = parse_args()
    walls_path = resolve_project_path(args.walls)
    hazards_path = resolve_project_path(args.hazards)
    knowledge_path = resolve_project_path(args.knowledge)
    display_dynamic_fire = not args.static_fire

    replay = build_replay(
        walls_path,
        hazards_path,
        knowledge_path,
        args.run,
        force_dynamic_fire=display_dynamic_fire,
    )
    output_path = (
        resolve_project_path(args.output)
        if args.output
        else VISUALIZER_DIR / f"pygame_replay_run{replay['run_number']}.gif"
    )
    render_meta = render_replay(
        replay,
        output_path,
        max_frames=args.max_frames,
        cell_px=args.cell_px,
        frame_duration_ms=args.duration,
    )

    if not args.no_latest_alias:
        latest_alias = VISUALIZER_DIR / "pygame_replay_latest.gif"
        if latest_alias.resolve() != output_path.resolve():
            latest_alias.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output_path, latest_alias)

    print(f"Saved GIF → {output_path}")
    print(f"Run          : {replay['run_number']}")
    print(f"Frames       : {render_meta['frame_count']}")
    print(f"Frame stride : {render_meta['sample_stride']}")
    print(f"Run hazards  : {replay['summary']['hazard_model']}")
    print(f"Display haz. : {replay['summary']['display_hazard_model']}")
    print(f"Fatal hits   : {replay['summary']['fatal_fire_hits']}")
    print(f"On-fire now  : {replay['summary']['on_fire_now']}")
    print(f"Min fire dst : {replay['summary']['min_fire_distance']}")


if __name__ == "__main__":
    main()
