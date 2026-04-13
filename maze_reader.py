"""
maze_reader.py
==============
Loads and inspects a maze image pair:
  - MAZE_0.png  →  wall layout
  - MAZE_1.png  →  hazard overlay

Public API
----------
load_maze(path)                           → (image_array, h_walls, v_walls)
find_start_goal(h_walls)                  → (start, goal)
get_start(h_walls)                        → (row, col)
get_goal(h_walls)                         → (row, col)

load_hazards(path)                        → {(row, col): Hazard}
print_summary(start, goal, hazards)       → prints maze summary

cell_center(row, col)                     → (x, y) pixel coords of cell centre

# Movement
in_bounds(row, col)                       → bool
can_move(row, col, direction, ...)        → bool

# Hazards (simplified)
if_alive(row, col, hazards)               → bool
get_hazard(row, col, hazards)             → Hazard | None

# Dynamic fire
update_fire_in_hazards(hazards)           → updated hazards dict
"""

from collections import Counter
from enum import Enum

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------
GRID  = 64   # cells per side
WALL  = 2    # wall-strip width in pixels
STEP  = 16   # pixels per cell (wall shared between neighbours)
INNER = 14   # usable inner-cell size in pixels

# ---------------------------------------------------------------------------
# Actions & directions
# ---------------------------------------------------------------------------
ACTION_DELTAS   = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
REVERSE_ACTION  = {0: 2, 1: 3, 2: 0, 3: 1}
ACTION_NAMES    = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

# ---------------------------------------------------------------------------
# Hazard definitions
# ---------------------------------------------------------------------------
class Hazard(Enum):
    FIRE       = "fire"
    CONFUSION  = "confusion"
    TP_GREEN   = "tp_green"
    TP_YELLOW  = "tp_yellow"
    TP_PURPLE  = "tp_purple"
    TP_RED     = "tp_red"

HAZARD_LABELS = {
    Hazard.FIRE:      "FIRE",
    Hazard.CONFUSION: "CONFUSION",
    Hazard.TP_GREEN:  "TP_GREEN",
    Hazard.TP_YELLOW: "TP_YELLOW",
    Hazard.TP_PURPLE: "TP_PURPLE",
    Hazard.TP_RED:     "TP_RED",
}

HAZARD_LABELS = {
    Hazard.FIRE:      "FIRE",
    Hazard.CONFUSION: "CONFUSION",
    Hazard.TP_GREEN:  "TP_GREEN",
    Hazard.TP_YELLOW: "TP_YELLOW",
    Hazard.TP_PURPLE: "TP_PURPLE",
    Hazard.TP_RED:     "TP_RED",
}

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def cell_center(row, col):
    """Return pixel (x, y) at the centre of cell (row, col)."""
    x = WALL + col * STEP + INNER // 2
    y = WALL + row * STEP + INNER // 2
    return x, y


# ---------------------------------------------------------------------------
# Wall loading
# ---------------------------------------------------------------------------
def load_maze(path="MAZE_0.png"):
    image = np.array(Image.open(path).convert("RGB"))
    black = np.zeros(3, dtype=np.uint8)

    v_walls = np.full((GRID, GRID + 1), False)
    h_walls = np.full((GRID + 1, GRID), False)

    for row in range(GRID):
        y0 = WALL + row * STEP
        y1 = y0 + INNER
        for wi in range(GRID + 1):
            strip = image[y0:y1, wi * STEP: wi * STEP + WALL]
            v_walls[row, wi] = np.mean(np.all(strip == black, axis=2)) > 0.5

    for wi in range(GRID + 1):
        y0, y1 = wi * STEP, wi * STEP + WALL
        for col in range(GRID):
            x0 = WALL + col * STEP
            strip = image[y0:y1, x0: x0 + INNER]
            h_walls[wi, col] = np.mean(np.all(strip == black, axis=2)) > 0.5

    return image, h_walls, v_walls


def find_start_goal(h_walls):
    """
    Find the open entry gap (bottom edge) and exit gap (top edge).

    Returns
    -------
    start : (row, col)
    goal  : (row, col)
    """
    start = (GRID - 1, int(np.where(~h_walls[-1])[0][0]))
    goal  = (0,        int(np.where(~h_walls[0])[0][0]))
    return start, goal


# ---------------------------------------------------------------------------
# Hazard loading
# ---------------------------------------------------------------------------
def _classify_color(r, g, b):
    """Map an average RGB value to a Hazard type, or None if unrecognised."""

    # confusion
    if 155 <= r <= 195 and 110 <= g <= 145 and 55 <= b <= 82:
        return Hazard.CONFUSION

    # fire
    if r >= 220 and 85 <= g <= 160 and 20 <= b <= 100 and (r - g) > 70:
        return Hazard.FIRE

    # green teleporter
    if g > 170 and b > 100 and r < 170:
        return Hazard.TP_GREEN

    # yellow / orange teleporter
    if r > 180 and g > 130 and b < 110:
        return Hazard.TP_YELLOW

    # purple teleporter
    if r > 90 and r < 210 and g < 150 and b > 140:
        return Hazard.TP_PURPLE
    
    if r > 200 and g < 100 and b < 100:
        return Hazard.TP_RED

    return None


def load_hazards(path="MAZE_1.png"):
    """
    Detect hazard type for every cell by sampling a 10×10 pixel patch
    around the cell centre, filtering out background pixels, then
    classifying the remaining colour.

    Returns
    -------
    hazards : dict  {(row, col): Hazard}
    """
    img = np.array(Image.open(path).convert("RGB"))
    hazards = {}

    for row in range(GRID):
        for col in range(GRID):
            cx, cy = cell_center(row, col)
            y0 = max(0, cy - 5);  y1 = min(img.shape[0], cy + 5)
            x0 = max(0, cx - 5);  x1 = min(img.shape[1], cx + 5)
            patch = img[y0:y1, x0:x1]

            is_white = np.all(patch > 228, axis=2)
            is_black = np.all(patch < 40,  axis=2)
            mask     = ~is_white & ~is_black
            if mask.sum() < 4:
                continue

            px      = patch[mask]
            r, g, b = float(px[:, 0].mean()), float(px[:, 1].mean()), float(px[:, 2].mean())

            hz = _classify_color(r, g, b)
            if hz is not None:
                hazards[(row, col)] = hz

    return hazards

# ---------------------------------------------------------------------------
# Teleportation Matcher
# ---------------------------------------------------------------------------
def get_teleport_points(hazards):
    """
    Return teleporter cells grouped by color.
    """
    return {
        Hazard.TP_GREEN: sorted([cell for cell, hz in hazards.items() if hz == Hazard.TP_GREEN]),
        Hazard.TP_YELLOW: sorted([cell for cell, hz in hazards.items() if hz == Hazard.TP_YELLOW]),
        Hazard.TP_PURPLE: sorted([cell for cell, hz in hazards.items() if hz == Hazard.TP_PURPLE]),
        Hazard.TP_RED : sorted([cell for cell, hz in hazards.items() if hz == Hazard.TP_RED]),
    }


def get_teleport_pairs(hazards):
    """
    Return teleporter pairs by color.
    Assumes each color has exactly 2 cells.
    """
    teleport_points = get_teleport_points(hazards)
    pairs = {}

    for color, cells in teleport_points.items():
        if len(cells) == 2:
            pairs[color] = (cells[0], cells[1])
        else:
            pairs[color] = tuple(cells)

    return pairs


def print_teleport_pairs_exact(hazards):
    pairs = get_teleport_pairs(hazards)

    print("\nTELEPORTER PAIRS")
    print("=" * 55)
    for color, pair in pairs.items():
        print(f"{color.value}: {pair}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Dynamic fire helpers
# ---------------------------------------------------------------------------
def find_fire_groups(fire_cells):
    """
    Split fire cells into connected groups.

    Uses 8-direction connectivity so diagonal V-shapes stay together.

    Returns
    -------
    groups : list[set[(row, col)]]
    """
    fire_cells = set(fire_cells)
    groups = []

    while fire_cells:
        start = fire_cells.pop()
        group = {start}
        stack = [start]

        while stack:
            row, col = stack.pop()

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue

                    nbr = (row + dr, col + dc)
                    if nbr in fire_cells:
                        fire_cells.remove(nbr)
                        group.add(nbr)
                        stack.append(nbr)

        groups.append(group)

    return groups


def find_fire_pivot(fire_group):
    """
    Find the pivot of one fire group.

    Strategy:
    choose the cell with the smallest total Manhattan distance
    to all other cells in the same group.
    """
    if not fire_group:
        return None

    best_cell = None
    best_score = float("inf")

    for r1, c1 in fire_group:
        score = 0
        for r2, c2 in fire_group:
            score += abs(r1 - r2) + abs(c1 - c2)

        if score < best_score:
            best_score = score
            best_cell = (r1, c1)

    return best_cell


def rotate_point_around_pivot_cw(row, col, pivot_row, pivot_col):
    """
    Rotate one cell 90 degrees clockwise around a pivot.
    """
    rel_row = row - pivot_row
    rel_col = col - pivot_col

    # 90 deg clockwise: (r, c) -> (c, -r)
    new_rel_row = rel_col
    new_rel_col = -rel_row

    new_row = pivot_row + new_rel_row
    new_col = pivot_col + new_rel_col

    return new_row, new_col


def rotate_fire_group_cw(fire_group, pivot, grid_size=GRID):
    """
    Rotate one fire group 90 degrees clockwise around its pivot.
    Cells outside the grid are removed.
    """
    pivot_row, pivot_col = pivot
    new_group = set()

    for row, col in fire_group:
        nr, nc = rotate_point_around_pivot_cw(row, col, pivot_row, pivot_col)

        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            new_group.add((nr, nc))

    return new_group


def update_fire_in_hazards(hazards, grid_size=GRID):
    """
    Rotate each separate fire group around its own auto-detected pivot.
    Non-fire hazards stay the same.
    """
    static_hazards = {cell: hz for cell, hz in hazards.items() if hz != Hazard.FIRE}
    fire_cells = {cell for cell, hz in hazards.items() if hz == Hazard.FIRE}

    fire_groups = find_fire_groups(fire_cells)
    new_fire_cells = set()

    for group in fire_groups:
        pivot = find_fire_pivot(group)

        if pivot is not None:
            rotated_group = rotate_fire_group_cw(group, pivot, grid_size)
            new_fire_cells.update(rotated_group)

    new_hazards = dict(static_hazards)
    for cell in new_fire_cells:
        new_hazards[cell] = Hazard.FIRE

    return new_hazards


# ---------------------------------------------------------------------------
# Query helpers / simple API
# ---------------------------------------------------------------------------
def in_bounds(row, col):
    """Return True if (row, col) is a valid maze cell."""
    return 0 <= row < GRID and 0 <= col < GRID


def can_move(row, col, direction, h_walls, v_walls):
    dr, dc = 0, 0

    if direction == "up":
        dr, dc = -1, 0
    elif direction == "down":
        dr, dc = 1, 0
    elif direction == "left":
        dr, dc = 0, -1
    elif direction == "right":
        dr, dc = 0, 1
    else:
        return False

    new_row = row + dr
    new_col = col + dc

    if not in_bounds(new_row, new_col):
        return False

    # wall check
    if dr == -1:  # up
        return not h_walls[row, col]
    if dr == 1:   # down
        return not h_walls[row + 1, col]
    if dc == -1:  # left
        return not v_walls[row, col]
    if dc == 1:   # right
        return not v_walls[row, col + 1]

    return False

def if_alive(row, col, hazards):
    return hazards.get((row, col)) != Hazard.FIRE


def get_hazard(row, col, hazards):
    return hazards.get((row, col))


def get_start(h_walls):
    start, _ = find_start_goal(h_walls)
    return start


def get_goal(h_walls):
    _, goal = find_start_goal(h_walls)
    return goal

# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(h_walls, hazards):
    start = get_start(h_walls)
    goal = get_goal(h_walls)

    print("\n" + "=" * 55)
    print("MAZE SUMMARY")
    print("=" * 55)
    print(f"Start : {start}")
    print(f"Goal  : {goal}")

    print(f"\nHazards detected: {len(hazards)}")

    counts = Counter(hazards.values())

    print(f"Fire cells      : {counts.get(Hazard.FIRE, 0)}")
    print(f"Confusion traps : {counts.get(Hazard.CONFUSION, 0)}")
    print(f"Green TP        : {counts.get(Hazard.TP_GREEN, 0)}")
    print(f"Yellow TP       : {counts.get(Hazard.TP_YELLOW, 0)}")
    print(f"Purple TP       : {counts.get(Hazard.TP_PURPLE, 0)}")
    print(f"Red TP          : {counts.get(Hazard.TP_RED, 0)}")

    print("=" * 55)

    print_teleport_pairs_exact(hazards)


# ---------------------------------------------------------------------------
# Entry point – run as script for a quick inspection
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading maze walls from MAZE_0.png...")
    image, h_walls, v_walls = load_maze("MAZE_0.png")

    print("Loading hazards from MAZE_1.png...")
    hazards = load_hazards("MAZE_1.png")

    print_summary(h_walls, hazards)