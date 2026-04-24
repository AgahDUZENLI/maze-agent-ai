from pathlib import Path
from PIL import Image, ImageDraw
import os
import argparse

from maze_reader import (
    GRID,
    load_maze,
    find_start_goal,
    load_hazards,
    Hazard,
    update_fire_in_hazards,
    init_fire_groups
)

CELL = 12  # small = fast rendering

COLORS = {
    "bg": (255, 255, 255),
    "wall": (0, 0, 0),
    "start": (0, 200, 0),
    "goal": (200, 0, 0),

    Hazard.FIRE:      (255, 120, 80),
    Hazard.CONFUSION: (120, 200, 255),
    Hazard.TP_GREEN:  (120, 255, 150),
    Hazard.TP_YELLOW: (255, 230, 100),
    Hazard.TP_PURPLE: (200, 120, 255),
    Hazard.TP_RED:    (255, 120, 120),
}


def render_map(h_walls, v_walls, start, goal, hazards, path):
    size = GRID * CELL
    img = Image.new("RGB", (size, size), COLORS["bg"])
    draw = ImageDraw.Draw(img)

    # draw cells
    for r in range(GRID):
        for c in range(GRID):
            x0 = c * CELL
            y0 = r * CELL
            x1 = x0 + CELL
            y1 = y0 + CELL

            if (r, c) == start:
                color = COLORS["start"]
            elif (r, c) == goal:
                color = COLORS["goal"]
            elif (r, c) in hazards:
                color = COLORS.get(hazards[(r, c)], (200, 200, 200))
            else:
                continue

            draw.rectangle([x0, y0, x1, y1], fill=color)

    # draw walls
    for r in range(GRID):
        for c in range(GRID):
            if h_walls[r, c]:
                draw.line([(c*CELL, r*CELL), ((c+1)*CELL, r*CELL)], fill=COLORS["wall"])
            if v_walls[r, c]:
                draw.line([(c*CELL, r*CELL), (c*CELL, (r+1)*CELL)], fill=COLORS["wall"])

    img.save(path)


def render_turns(h_walls, v_walls, start, goal, hazards, out_dir, steps=8):
    current = dict(hazards)
    state = init_fire_groups(current)

    for t in range(steps + 1):
        print(f"Turn {t}")

        render_map(
            h_walls, v_walls, start, goal,
            current,
            out_dir / f"turn_{t}.png"
        )

        current, state = update_fire_in_hazards(current, state)


# ---------------------------------------------------------------------------
# MAIN (FIXED)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maze", "-m",
        choices=["alpha", "beta", "gamma"],
        default="alpha"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=5,
        help="Number of simulation steps"
    )
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MAZE_DIR  = os.path.join(BASE_DIR, "TestMazes", f"maze-{args.maze}")
    MAZE_PATH = os.path.join(MAZE_DIR, "MAZE_0.png")
    HAZ_PATH  = os.path.join(MAZE_DIR, "MAZE_1.png")

    out_dir = Path(BASE_DIR) / "results" / args.maze
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading maze: {MAZE_PATH}")
    print(f"Loading hazards: {HAZ_PATH}")

    if not os.path.exists(MAZE_PATH):
        raise FileNotFoundError(MAZE_PATH)
    if not os.path.exists(HAZ_PATH):
        raise FileNotFoundError(HAZ_PATH)

    _, h_walls, v_walls = load_maze(MAZE_PATH)
    start, goal = find_start_goal(h_walls)
    hazards = load_hazards(HAZ_PATH)

    render_turns(
        h_walls,
        v_walls,
        start,
        goal,
        hazards,
        out_dir,
        steps=args.steps
    )

    print("\nDone.")