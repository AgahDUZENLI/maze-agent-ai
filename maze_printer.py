from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from maze_reader import (
    GRID,
    load_maze,
    find_start_goal,
    load_hazards,
    Hazard,
    HAZARD_LABELS,
    update_fire_in_hazards,
)

# ---------------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------------
CELL_PX = 100

HAZARD_FILL = {
    Hazard.FIRE:      (255, 180, 120),
    Hazard.CONFUSION: (160, 230, 255),
    Hazard.TP_GREEN:  (120, 230, 150),
    Hazard.TP_YELLOW: (255, 235, 100),
    Hazard.TP_PURPLE: (210, 140, 255),
    Hazard.TP_RED:    (255, 150, 150),
    Hazard.PUSH_UP:   (70, 130, 180),
    Hazard.PUSH_LEFT: (100, 149, 237),
}

HAZARD_SHORT = {
    Hazard.FIRE:      "FIRE",
    Hazard.CONFUSION: "CONF",
    Hazard.TP_GREEN:  "GRN",
    Hazard.TP_YELLOW: "YLW",
    Hazard.TP_PURPLE: "PRP",
    Hazard.TP_RED:    "RED",
    Hazard.PUSH_UP:   "P↑",
    Hazard.PUSH_LEFT: "P←",
}

START_FILL = (180, 255, 180)
GOAL_FILL  = (255, 160, 160)
WALL_COL   = (30, 30, 30)
BG_COL     = (255, 255, 255)

COORD_COL  = (80, 80, 80)
HAZ_COL    = (20, 20, 20)


# ---------------------------------------------------------------------------
# Font loader
# ---------------------------------------------------------------------------
def _font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            size
        )
    except Exception:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Fire debug printer
# ---------------------------------------------------------------------------
def print_fire_turns(hazards, num_turns=4):
    current_hazards = dict(hazards)
    pivots = None

    for turn in range(num_turns + 1):
        fire_cells = sorted(
            [cell for cell, hz in current_hazards.items() if hz == Hazard.FIRE]
        )

        print(f"\nTURN {turn}")
        print(f"Fire cells ({len(fire_cells)}):")
        for cell in fire_cells:
            print(cell)

        if turn < num_turns:
            current_hazards, pivots = update_fire_in_hazards(current_hazards, pivots)


# ---------------------------------------------------------------------------
# Annotated renderer
# ---------------------------------------------------------------------------
def render_annotated(h_walls, v_walls, start, goal, hazards, out_path, turn=None):
    wall_px = max(2, CELL_PX // 10)
    total = GRID * CELL_PX + wall_px

    img = Image.new("RGB", (total, total), BG_COL)
    draw = ImageDraw.Draw(img)

    coord_font = _font(max(8, CELL_PX // 9))
    haz_font = _font(max(9, CELL_PX // 8))
    title_font = _font(22)

    # draw cell fills
    for row in range(GRID):
        for col in range(GRID):
            x0 = col * CELL_PX + wall_px
            y0 = row * CELL_PX + wall_px
            x1 = x0 + CELL_PX - wall_px
            y1 = y0 + CELL_PX - wall_px

            if (row, col) == start:
                fill = START_FILL
            elif (row, col) == goal:
                fill = GOAL_FILL
            elif (row, col) in hazards:
                fill = HAZARD_FILL.get(hazards[(row, col)], (230, 230, 230))
            else:
                fill = BG_COL

            draw.rectangle([x0, y0, x1, y1], fill=fill)

    # draw horizontal walls
    for wi in range(GRID + 1):
        for col in range(GRID):
            if h_walls[wi, col]:
                x0 = col * CELL_PX
                y0 = wi * CELL_PX
                x1 = x0 + CELL_PX + wall_px
                y1 = y0 + wall_px
                draw.rectangle([x0, y0, x1, y1], fill=WALL_COL)

    # draw vertical walls
    for row in range(GRID):
        for wi in range(GRID + 1):
            if v_walls[row, wi]:
                x0 = wi * CELL_PX
                y0 = row * CELL_PX
                x1 = x0 + wall_px
                y1 = y0 + CELL_PX + wall_px
                draw.rectangle([x0, y0, x1, y1], fill=WALL_COL)

    # labels
    for row in range(GRID):
        for col in range(GRID):
            cx = col * CELL_PX + CELL_PX // 2
            cy = row * CELL_PX + CELL_PX // 2

            coord_text = f"({row},{col})"
            bbox = draw.textbbox((0, 0), coord_text, font=coord_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (cx - tw // 2, cy - CELL_PX // 4 - th // 2),
                coord_text,
                fill=COORD_COL,
                font=coord_font
            )

            if (row, col) == start:
                label = "START"
            elif (row, col) == goal:
                label = "GOAL"
            elif (row, col) in hazards:
                label = HAZARD_SHORT[hazards[(row, col)]]
            else:
                label = None

            if label:
                bbox2 = draw.textbbox((0, 0), label, font=haz_font)
                lw = bbox2[2] - bbox2[0]
                draw.text(
                    (cx - lw // 2, cy + CELL_PX // 8),
                    label,
                    fill=HAZ_COL,
                    font=haz_font
                )

    if turn is not None:
        draw.text((10, 10), f"TURN {turn}", fill=(0, 0, 0), font=title_font)

    img.save(out_path)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Render every fire turn
# ---------------------------------------------------------------------------
def render_fire_turns(h_walls, v_walls, start, goal, hazards, out_dir, num_turns=4):
    current_hazards = dict(hazards)
    pivots = None

    for turn in range(num_turns + 1):
        out_path = out_dir / f"maze_turn_{turn}.png"
        render_annotated(
            h_walls, v_walls, start, goal,
            current_hazards, out_path, turn=turn
        )

        fire_cells = sorted(
            [cell for cell, hz in current_hazards.items() if hz == Hazard.FIRE]
        )
        print(f"\nTURN {turn}")
        print(f"Fire count: {len(fire_cells)}")
        print(fire_cells)

        if turn < num_turns:
            current_hazards, pivots = update_fire_in_hazards(current_hazards, pivots)


# ---------------------------------------------------------------------------
# Legend image
# ---------------------------------------------------------------------------
def render_legend(hazards, start, goal, out_path):
    entries = [
        ("START", START_FILL, "Entry cell"),
        ("GOAL", GOAL_FILL, "Exit cell"),
    ] + [
        (HAZARD_SHORT[h], HAZARD_FILL.get(h, (200, 200, 200)), HAZARD_LABELS[h])
        for h in sorted({v for v in hazards.values()}, key=lambda x: x.value)
    ]

    row_h = 40
    sw = 30
    width = 320
    height = row_h * len(entries) + 20
    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    font = _font(15)

    draw.text((8, 4), "LEGEND", fill=(0, 0, 0), font=_font(16))

    for i, (short, fill, label) in enumerate(entries):
        y = 24 + i * row_h
        draw.rectangle([8, y, 8 + sw, y + sw], fill=fill, outline=(80, 80, 80))
        draw.text(
            (8 + sw + 8, y + 6),
            f"{short}  —  {label}",
            fill=(20, 20, 20),
            font=font
        )

    img.save(out_path)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print("Loading maze...")
    image, h_walls, v_walls = load_maze("MAZE_0.png")
    start, goal = find_start_goal(h_walls)
    print(f"start={start}  goal={goal}")

    print("Loading hazards...")
    hazards = load_hazards("MAZE_1.png")
    print(f"{len(hazards)} hazard cells found")

    print("\nPrinting fire turns...")
    print_fire_turns(hazards, num_turns=8)

    print("\nRendering original annotated maze...")
    render_annotated(
        h_walls, v_walls, start, goal, hazards,
        out_path=out_dir / "annotated_maze.png"
    )

    render_legend(
        hazards, start, goal,
        out_path=out_dir / "annotated_maze_legend.png"
    )

    print("\nRendering fire turns...")
    render_fire_turns(
        h_walls, v_walls, start, goal, hazards,
        out_dir=out_dir,
        num_turns=8
    )

    print("\nDone. Files in results/:")
    print("  annotated_maze.png")
    print("  annotated_maze_legend.png")
    print("  maze_turn_0.png ... maze_turn_8.png")