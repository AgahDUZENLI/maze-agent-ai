import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

PROJECT_DIR = Path(__file__).resolve().parent

from maze_reader import GRID, Hazard, init_fire_groups, load_hazards, load_maze, update_fire_in_hazards


FPS = 24
DEFAULT_CELL_SIZE = 12
DEFAULT_MAX_FRAMES = 240
EVENT_HOLD_SECONDS = 2.0
FRAME_PAD = 24
HEADER_H = 72
FOOTER_H = 0

BG = (12, 17, 23)
FRAME_BG = (20, 26, 35)
FRAME_BORDER = (63, 74, 90)
HEADER_BG = (244, 239, 229)
HEADER_TEXT = (28, 31, 38)
SUBTEXT = (104, 112, 125)
BOARD_BG = (14, 19, 27)
BOARD_TILE_A = (37, 47, 61)
BOARD_TILE_B = (33, 42, 55)
GRID_LINE = (50, 61, 76)
WALL = (232, 236, 242)
START = (74, 190, 134)
GOAL = (224, 98, 103)
VISITED = (128, 185, 255, 34)
VISITED_LINE = (146, 199, 255, 54)
PATH_GLOW = (120, 181, 255, 72)
PATH = (120, 181, 255, 190)
PATH_TAIL = (255, 233, 191, 245)
CURRENT = (255, 255, 255)
CURRENT_RING = (132, 190, 255)
ACCENT = (118, 192, 255)
FIRE = (255, 134, 74, 135)
FIRE_CORE = (255, 205, 130, 165)
CONFUSION = (98, 218, 255, 72)
CONFUSION_LINE = (98, 218, 255)
TELEPORT = (199, 142, 255, 72)
TELEPORT_LINE = (199, 142, 255)
FAIL = (242, 110, 110)
WARN = (246, 230, 199)
SUCCESS = (220, 239, 232)
CHIP_LIGHT = (227, 234, 244)
CHIP_FAIL = (245, 220, 220)


@dataclass
class MazeScenario:
    name: str
    maze_path: Path
    hazard_path: Path
    start: Tuple[int, int]
    goal: Tuple[int, int]
    h_walls: object
    v_walls: object
    base_hazards: Dict[Tuple[int, int], Hazard]


@dataclass
class ReplayFrame:
    index: int
    current: Tuple[int, int]
    visited: Tuple[Tuple[int, int], ...]
    path: Tuple[Tuple[int, int], ...]
    deaths: Tuple[Tuple[int, int], ...]
    status: str
    label: str
    hazards: Dict[Tuple[int, int], Hazard]


@dataclass
class EventOverlay:
    text: str
    fill: Tuple[int, int, int]
    text_color: Tuple[int, int, int]
    frames_left: int
    total_frames: int


def resolve_replay_asset_path(path_str: str, replay_path: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    replay_candidates = [
        replay_path.parent / path,
        replay_path.parent.parent / path,
        replay_path.parent.parent.parent / path,
    ]
    for candidate in replay_candidates:
        if candidate.exists():
            return candidate.resolve()

    return (PROJECT_DIR / path).resolve()


def load_replay(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        replay = json.load(f)
    replay["_source_path"] = path.resolve()
    return replay


def load_scenario_from_replay(replay: dict) -> MazeScenario:
    source_path = Path(replay["_source_path"])
    maze_path = resolve_replay_asset_path(replay["maze_path"], source_path)
    hazard_path = resolve_replay_asset_path(replay["hazard_path"], source_path)
    _, h_walls, v_walls = load_maze(str(maze_path))
    base_hazards = load_hazards(str(hazard_path))
    return MazeScenario(
        name=maze_path.parent.name if maze_path.parent.name.startswith("maze-") else maze_path.stem,
        maze_path=maze_path,
        hazard_path=hazard_path,
        start=tuple(replay["start"]),
        goal=tuple(replay["goal"]),
        h_walls=h_walls,
        v_walls=v_walls,
        base_hazards=base_hazards,
    )


def event_status_and_label(event: dict) -> Tuple[str, str]:
    action_names = {0: "up", 1: "down", 2: "left", 3: "right", 4: "wait"}

    if event.get("goal_reached"):
        return "goal", "goal"
    if event.get("died"):
        return "death", "death"
    if event.get("respawn_to_start"):
        return "respawn", "respawn"
    if event.get("teleported"):
        return "teleport", "teleport"
    if event.get("confused"):
        return "confused", "confusion"
    if event.get("wall_hit"):
        return "wall", "wall hit"
    if event.get("fire_rotated"):
        return "fire", "fire shift"

    submitted = action_names.get(event.get("submitted_action"), "n/a")
    effective = action_names.get(event.get("effective_action"), "n/a")
    if submitted == effective:
        return "move", submitted
    return "move", f"{submitted}->{effective}"


def build_replay_frames(replay: dict, scenario: MazeScenario) -> List[ReplayFrame]:
    frames: List[ReplayFrame] = []
    events = replay.get("events") or []
    current_hazards = dict(scenario.base_hazards)
    fire_pivots = init_fire_groups(current_hazards)
    path: List[Tuple[int, int]] = [scenario.start]
    visited = {scenario.start}
    deaths: List[Tuple[int, int]] = []

    frames.append(
        ReplayFrame(
            index=0,
            current=scenario.start,
            visited=(scenario.start,),
            path=(scenario.start,),
            deaths=(),
            status="start",
            label="start",
            hazards=dict(current_hazards),
        )
    )

    for event in events:
        if event.get("fire_rotated"):
            current_hazards, fire_pivots = update_fire_in_hazards(current_hazards, fire_pivots)

        pos = tuple(event["position"])
        path.append(pos)
        visited.add(pos)
        if event.get("died"):
            deaths.append(pos)

        status, label = event_status_and_label(event)
        frames.append(
            ReplayFrame(
                index=len(frames),
                current=pos,
                visited=tuple(sorted(visited)),
                path=tuple(path),
                deaths=tuple(deaths),
                status=status,
                label=label,
                hazards=dict(current_hazards),
            )
        )

        if event.get("respawn_to_start"):
            path.append(scenario.start)
            visited.add(scenario.start)
            frames.append(
                ReplayFrame(
                    index=len(frames),
                    current=scenario.start,
                    visited=tuple(sorted(visited)),
                    path=tuple(path),
                    deaths=tuple(deaths),
                    status="respawn",
                    label="respawn",
                    hazards=dict(current_hazards),
                )
            )

    return frames


def evenly_sample(items: Sequence[int], count: int) -> List[int]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]

    chosen = []
    last = None
    for i in range(count):
        idx = round(i * (len(items) - 1) / (count - 1))
        if idx != last:
            chosen.append(items[idx])
            last = idx
    if len(chosen) < count:
        for item in items:
            if item not in chosen:
                chosen.append(item)
            if len(chosen) == count:
                break
    return chosen[:count]


def is_special_frame(frame: ReplayFrame) -> bool:
    return frame.status in {"goal", "death", "respawn", "teleport", "confused", "wall", "fire"}


def select_frame_indices(frames: Sequence[ReplayFrame], max_frames: int) -> List[int]:
    if not frames:
        return []
    if len(frames) <= max_frames:
        return list(range(len(frames)))

    mandatory = {0, len(frames) - 1}
    for idx, frame in enumerate(frames):
        if is_special_frame(frame):
            for delta in (-1, 0, 1):
                probe = idx + delta
                if 0 <= probe < len(frames):
                    mandatory.add(probe)

    mandatory = sorted(mandatory)
    if len(mandatory) >= max_frames:
        middle = mandatory[1:-1]
        keep = [mandatory[0], *evenly_sample(middle, max_frames - 2), mandatory[-1]]
        return sorted(set(keep))

    chosen = set(mandatory)
    remaining = [idx for idx in range(len(frames)) if idx not in chosen]
    chosen.update(evenly_sample(remaining, max_frames - len(chosen)))
    return sorted(chosen)


def board_cell_rect(col: int, row: int, cell_size: int, origin_x: int, origin_y: int) -> pygame.Rect:
    return pygame.Rect(origin_x + col * cell_size, origin_y + row * cell_size, cell_size, cell_size)


def board_center(cell: Tuple[int, int], cell_size: int, origin_x: int, origin_y: int) -> Tuple[int, int]:
    row, col = cell
    return (
        origin_x + col * cell_size + cell_size // 2,
        origin_y + row * cell_size + cell_size // 2,
    )


def draw_text(surface, font, text, color, pos):
    label = font.render(text, True, color)
    surface.blit(label, pos)
    return label.get_rect(topleft=pos)


def draw_chip(surface, font, x, y, text, fill, text_color):
    label = font.render(text, True, text_color)
    rect = pygame.Rect(x, y, label.get_width() + 18, 24)
    pygame.draw.rect(surface, fill, rect, border_radius=12)
    surface.blit(label, (rect.x + 9, rect.y + 4))
    return rect


def draw_chip_right(surface, font, right_x, y, text, fill, text_color):
    label = font.render(text, True, text_color)
    rect = pygame.Rect(right_x - label.get_width() - 18, y, label.get_width() + 18, 24)
    pygame.draw.rect(surface, fill, rect, border_radius=12)
    surface.blit(label, (rect.x + 9, rect.y + 4))
    return rect


def draw_chip_right_alpha(surface, font, right_x, y, text, fill, text_color, alpha):
    label = font.render(text, True, text_color)
    width = label.get_width() + 18
    height = 24
    chip = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(chip, (*fill, alpha), pygame.Rect(0, 0, width, height), border_radius=12)
    label.set_alpha(alpha)
    chip.blit(label, (9, 4))
    rect = chip.get_rect(topright=(right_x, y))
    surface.blit(chip, rect.topleft)
    return rect


def draw_chip_alpha(surface, font, x, y, text, fill, text_color, alpha):
    label = font.render(text, True, text_color)
    width = label.get_width() + 18
    height = 24
    chip = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(chip, (*fill, alpha), pygame.Rect(0, 0, width, height), border_radius=12)
    label.set_alpha(alpha)
    chip.blit(label, (9, 4))
    rect = chip.get_rect(topleft=(x, y))
    surface.blit(chip, rect.topleft)
    return rect


def frame_badge(frame: ReplayFrame):
    if frame.status == "goal":
        return "Goal", SUCCESS, HEADER_TEXT
    if frame.status == "death":
        return "Death", CHIP_FAIL, HEADER_TEXT
    if frame.status == "respawn":
        return "Respawn", WARN, HEADER_TEXT
    if frame.status == "teleport":
        return "Teleport", CHIP_LIGHT, HEADER_TEXT
    if frame.status == "confused":
        return "Confused", CHIP_LIGHT, HEADER_TEXT
    if frame.status == "wall":
        return "Wall", WARN, HEADER_TEXT
    if frame.status == "fire":
        return "Fire shift", WARN, HEADER_TEXT
    return "Replay", CHIP_LIGHT, HEADER_TEXT


def make_event_overlay(frame: ReplayFrame) -> Optional[EventOverlay]:
    if frame.status in {"start", "move"}:
        return None
    text, fill, text_color = frame_badge(frame)
    hold_frames = max(1, int(round(EVENT_HOLD_SECONDS * FPS)))
    return EventOverlay(
        text=text,
        fill=fill,
        text_color=text_color,
        frames_left=hold_frames,
        total_frames=hold_frames,
    )


def event_alpha(event: EventOverlay, position: int) -> int:
    life_ratio = event.frames_left / max(1, event.total_frames)
    base = 80 + int(160 * life_ratio)
    rank_penalty = position * 36
    return max(70, min(255, base - rank_penalty))


def draw_recent_events(surface, font, header_rect: pygame.Rect, events: Sequence[EventOverlay]):
    if not events:
        return

    displayed_events = list(reversed(events[:3]))
    chip_gap = 8
    chip_widths = [
        font.render(event.text, True, event.text_color).get_width() + 18
        for event in displayed_events
    ]
    total_width = sum(chip_widths) + chip_gap * max(0, len(chip_widths) - 1)
    chip_x = header_rect.right - 12 - total_width
    chip_y = header_rect.centery - 12

    for idx, event in enumerate(displayed_events):
        alpha = event_alpha(event, idx)
        draw_chip_alpha(surface, font, chip_x, chip_y, event.text, event.fill, event.text_color, alpha)
        chip_x += chip_widths[idx] + chip_gap


def draw_base(surface, scenario: MazeScenario, hazards: Dict[Tuple[int, int], Hazard], cell_size: int, origin_x: int, origin_y: int):
    board_w = GRID * cell_size
    board_h = GRID * cell_size
    board_rect = pygame.Rect(origin_x, origin_y, board_w, board_h)
    pygame.draw.rect(surface, BOARD_BG, board_rect, border_radius=18)

    for row in range(GRID):
        for col in range(GRID):
            rect = board_cell_rect(col, row, cell_size, origin_x, origin_y)
            color = BOARD_TILE_A if (row + col) % 2 == 0 else BOARD_TILE_B
            if (row, col) == scenario.start:
                color = START
            elif (row, col) == scenario.goal:
                color = GOAL
            pygame.draw.rect(surface, color, rect)

    for row in range(1, GRID):
        y = origin_y + row * cell_size
        pygame.draw.line(surface, GRID_LINE, (origin_x, y), (origin_x + board_w, y), 1)
    for col in range(1, GRID):
        x = origin_x + col * cell_size
        pygame.draw.line(surface, GRID_LINE, (x, origin_y), (x, origin_y + board_h), 1)

    for (row, col), hazard in hazards.items():
        rect = board_cell_rect(col, row, cell_size, origin_x, origin_y).inflate(-2, -2)
        if hazard == Hazard.FIRE:
            pygame.draw.rect(surface, FIRE, rect, border_radius=4)
            inner = rect.inflate(-max(2, cell_size // 3), -max(2, cell_size // 3))
            pygame.draw.ellipse(surface, FIRE_CORE, inner)
        elif hazard == Hazard.CONFUSION:
            pygame.draw.rect(surface, CONFUSION, rect, border_radius=4)
            pygame.draw.rect(surface, CONFUSION_LINE, rect, width=2, border_radius=4)
        else:
            pygame.draw.rect(surface, TELEPORT, rect, border_radius=4)
            pygame.draw.ellipse(surface, TELEPORT_LINE, rect, width=2)

    wall_width = 2 if cell_size < 14 else 3
    for wi in range(GRID + 1):
        for col in range(GRID):
            if scenario.h_walls[wi, col]:
                x0 = origin_x + col * cell_size
                y0 = origin_y + wi * cell_size
                pygame.draw.line(surface, WALL, (x0, y0), (x0 + cell_size, y0), wall_width)

    for row in range(GRID):
        for wi in range(GRID + 1):
            if scenario.v_walls[row, wi]:
                x0 = origin_x + wi * cell_size
                y0 = origin_y + row * cell_size
                pygame.draw.line(surface, WALL, (x0, y0), (x0, y0 + cell_size), wall_width)


def draw_deaths(surface, deaths: Sequence[Tuple[int, int]], cell_size: int, origin_x: int, origin_y: int):
    size = max(3, cell_size // 3)
    for row, col in deaths:
        cx, cy = board_center((row, col), cell_size, origin_x, origin_y)
        pygame.draw.line(surface, FAIL, (cx - size, cy - size), (cx + size, cy + size), 2)
        pygame.draw.line(surface, FAIL, (cx + size, cy - size), (cx - size, cy + size), 2)


def draw_path(overlay, path_cells: Sequence[Tuple[int, int]], cell_size: int):
    if len(path_cells) < 2:
        return
    points = [board_center(cell, cell_size, 0, 0) for cell in path_cells]
    pygame.draw.lines(overlay, PATH_GLOW, False, points, max(4, cell_size // 2))
    pygame.draw.lines(overlay, PATH, False, points, max(2, cell_size // 3))

    tail_points = points[max(0, len(points) - 10):]
    if len(tail_points) > 1:
        pygame.draw.lines(overlay, PATH_TAIL, False, tail_points, max(3, cell_size // 3))


def draw_current(surface, frame: ReplayFrame, cell_size: int, origin_x: int, origin_y: int, pulse: float):
    cx, cy = board_center(frame.current, cell_size, origin_x, origin_y)
    outer = max(5, cell_size // 2)
    pulse_radius = outer + 2 + int(3 * pulse)
    pygame.draw.circle(surface, CURRENT_RING, (cx, cy), pulse_radius, 2)
    pygame.draw.circle(surface, CURRENT, (cx, cy), outer)
    pygame.draw.circle(surface, ACCENT, (cx, cy), max(2, outer - 3))


def render_frame(
    scenario: MazeScenario,
    frame: ReplayFrame,
    frame_index: int,
    total_frames: int,
    cell_size: int,
    pulse_phase: int,
    recent_events: Sequence[EventOverlay],
):
    pulse = 0.25 + 0.5 * pulse_phase
    board_w = GRID * cell_size
    board_h = GRID * cell_size
    width = board_w + FRAME_PAD * 2
    height = HEADER_H + board_h + FOOTER_H + FRAME_PAD
    origin_x = FRAME_PAD
    origin_y = HEADER_H

    surface = pygame.Surface((width, height))
    surface.fill(BG)
    frame_rect = pygame.Rect(8, 8, width - 16, height - 16)
    pygame.draw.rect(surface, FRAME_BG, frame_rect, border_radius=26)
    pygame.draw.rect(surface, FRAME_BORDER, frame_rect, width=2, border_radius=26)
    header_rect = pygame.Rect(FRAME_PAD, 12, width - FRAME_PAD * 2, HEADER_H - 24)
    pygame.draw.rect(surface, HEADER_BG, header_rect, border_radius=12)

    draw_base(surface, scenario, frame.hazards, cell_size, origin_x, origin_y)

    overlay = pygame.Surface((board_w, board_h), pygame.SRCALPHA)
    for row, col in frame.visited:
        rect = pygame.Rect(col * cell_size + 1, row * cell_size + 1, cell_size - 2, cell_size - 2)
        pygame.draw.rect(overlay, VISITED, rect, border_radius=4)
        pygame.draw.rect(overlay, VISITED_LINE, rect, width=1, border_radius=4)

    draw_path(overlay, frame.path, cell_size)
    surface.blit(overlay, (origin_x, origin_y))
    draw_deaths(surface, frame.deaths, cell_size, origin_x, origin_y)
    draw_current(surface, frame, cell_size, origin_x, origin_y, pulse)

    header_font = pygame.font.SysFont("Avenir Next", 20)
    body_font = pygame.font.SysFont("Avenir Next", 16)

    title_rect = header_font.render(scenario.name, True, HEADER_TEXT).get_rect()
    title_y = header_rect.y + (header_rect.height - title_rect.height) // 2
    draw_text(surface, header_font, scenario.name, HEADER_TEXT, (header_rect.x + 12, title_y))

    draw_recent_events(surface, body_font, header_rect, recent_events)

    return surface


def write_video_from_frames(frame_paths: Sequence[Path], output_path: Path, fps: int):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(Path(frame_paths[0]).parent / "frame_%04d.png")
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def render_replay_to_video(replay_path: Path, output_path: Path, cell_size: int, max_frames: int):
    replay = load_replay(replay_path)
    scenario = load_scenario_from_replay(replay)
    frames = build_replay_frames(replay, scenario)
    selected_indices = select_frame_indices(frames, max_frames=max_frames)
    selected_frames = [frames[idx] for idx in selected_indices]

    with tempfile.TemporaryDirectory(prefix=f"{scenario.name}-replay-") as tmpdir:
        tmpdir = Path(tmpdir)
        frame_paths: List[Path] = []
        render_count = 0
        recent_events: List[EventOverlay] = []

        for logical_index, frame in enumerate(selected_frames):
            event_overlay = make_event_overlay(frame)
            if event_overlay is not None:
                recent_events.insert(0, event_overlay)
                recent_events = recent_events[:3]

            pulse_count = 3 if is_special_frame(frame) else 2
            for pulse_phase in range(pulse_count):
                surface = render_frame(
                    scenario=scenario,
                    frame=frame,
                    frame_index=logical_index,
                    total_frames=len(selected_frames),
                    cell_size=cell_size,
                    pulse_phase=pulse_phase,
                    recent_events=recent_events,
                )
                frame_path = tmpdir / f"frame_{render_count:04d}.png"
                pygame.image.save(surface, str(frame_path))
                frame_paths.append(frame_path)
                render_count += 1
                next_events: List[EventOverlay] = []
                for event in recent_events:
                    event.frames_left -= 1
                    if event.frames_left > 0:
                        next_events.append(event)
                recent_events = next_events

        if frame_paths:
            last_path = frame_paths[-1]
            for _ in range(FPS):
                hold_path = tmpdir / f"frame_{render_count:04d}.png"
                with open(last_path, "rb") as src, open(hold_path, "wb") as dst:
                    dst.write(src.read())
                frame_paths.append(hold_path)
                render_count += 1

            write_video_from_frames(frame_paths, output_path, FPS)


def default_replay_paths(test_root: Path, requested_mazes: Sequence[str]) -> List[Path]:
    if requested_mazes:
        maze_names = requested_mazes
    else:
        maze_names = sorted([path.name for path in test_root.iterdir() if path.is_dir() and path.name.startswith("maze-")])

    replays = []
    for maze_name in maze_names:
        replay_dir = test_root / maze_name / "results" / "replays"
        latest_path = replay_dir / "latest.json"
        preferred_path = None

        episode_paths = sorted(replay_dir.glob("episode_*.json"))
        for candidate in reversed(episode_paths):
            try:
                replay = load_replay(candidate)
            except Exception:
                continue
            if replay.get("summary", {}).get("goal_reached"):
                preferred_path = candidate
                break

        if preferred_path is not None:
            replays.append(preferred_path)
        elif latest_path.exists():
            replays.append(latest_path)
    return replays


def parse_args():
    parser = argparse.ArgumentParser(description="Render a styled video from solver replay JSON outputs.")
    parser.add_argument("--maze", action="append", default=[], help="Maze folder name inside TestMazes, e.g. maze-alpha.")
    parser.add_argument("--replay", action="append", default=[], help="Direct replay JSON path. Can be passed multiple times.")
    parser.add_argument("--all", action="store_true", help="Render all TestMazes replay outputs.")
    parser.add_argument("--output-dir", default=None, help='Output folder. Defaults to "maze-agent-ai/results/visualizer2".')
    parser.add_argument("--cell-size", type=int, default=DEFAULT_CELL_SIZE, help="Rendered cell size in pixels.")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="Maximum sampled logical replay frames.")
    return parser.parse_args()


def main():
    args = parse_args()
    pygame.init()
    pygame.font.init()

    output_root = Path(args.output_dir) if args.output_dir else PROJECT_DIR / "results" / "visualizer2"
    test_root = PROJECT_DIR / "TestMazes"

    replay_paths = [Path(path).resolve() for path in args.replay]
    if args.all or args.maze or not replay_paths:
        replay_paths.extend(default_replay_paths(test_root, args.maze))

    deduped = []
    seen = set()
    for path in replay_paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    replay_paths = deduped

    if not replay_paths:
        raise SystemExit(
            "No replay JSON files found. Expected solver outputs like "
            "'TestMazes/<maze>/results/replays/latest.json'."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    for replay_path in replay_paths:
        replay = load_replay(replay_path)
        scenario = load_scenario_from_replay(replay)
        output_path = output_root / f"{scenario.name}.mp4"
        render_replay_to_video(
            replay_path=replay_path,
            output_path=output_path,
            cell_size=args.cell_size,
            max_frames=args.max_frames,
        )
        print(f"[rendered] {replay_path} -> {output_path}")

    pygame.quit()


if __name__ == "__main__":
    main()
