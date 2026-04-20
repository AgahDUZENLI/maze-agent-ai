import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from maze_reader import (
    DEFAULT_HAZARDS_PATH,
    DEFAULT_WALLS_PATH,
    GRID,
    HAZARD_LABELS,
    Hazard,
    find_start_goal,
    load_hazards,
    load_maze,
    update_fire_in_hazards,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "printer"

CELL_PX = 100

HAZARD_FILL = {
    Hazard.FIRE: (255, 180, 120),
    Hazard.CONFUSION: (160, 230, 255),
    Hazard.TP_GREEN: (120, 230, 150),
    Hazard.TP_YELLOW: (255, 235, 100),
    Hazard.TP_PURPLE: (210, 140, 255),
    Hazard.TP_RED: (255, 150, 150),
}

HAZARD_SHORT = {
    Hazard.FIRE: "FIRE",
    Hazard.CONFUSION: "CONF",
    Hazard.TP_GREEN: "GRN",
    Hazard.TP_YELLOW: "YLW",
    Hazard.TP_PURPLE: "PRP",
    Hazard.TP_RED: "RED",
}

START_FILL = (180, 255, 180)
GOAL_FILL = (255, 160, 160)
WALL_COL = (30, 30, 30)
BG_COL = (255, 255, 255)
COORD_COL = (80, 80, 80)
HAZ_COL = (20, 20, 20)


def resolve_project_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def maze_label_from_paths(walls_path, hazards_path):
    walls_path = Path(walls_path)
    hazards_path = Path(hazards_path)
    if walls_path.parent != BASE_DIR:
        return walls_path.parent.name
    if hazards_path.parent != BASE_DIR:
        return hazards_path.parent.name
    return walls_path.stem.lower()


def default_output_dir(walls_path, hazards_path):
    return RESULTS_DIR / maze_label_from_paths(walls_path, hazards_path)


def _font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            size,
        )
    except Exception:
        return ImageFont.load_default()


def print_fire_turns(hazards, num_turns=4):
    current_hazards = dict(hazards)
    pivots = None

    for turn in range(num_turns + 1):
        fire_cells = sorted(
            [cell for cell, hazard in current_hazards.items() if hazard == Hazard.FIRE]
        )

        print(f"\nTURN {turn}")
        print(f"Fire cells ({len(fire_cells)}):")
        for cell in fire_cells:
            print(cell)

        if turn < num_turns:
            current_hazards, pivots = update_fire_in_hazards(current_hazards, pivots)


def render_annotated(h_walls, v_walls, start, goal, hazards, out_path, title=None):
    wall_px = max(2, CELL_PX // 10)
    total = GRID * CELL_PX + wall_px

    image = Image.new("RGB", (total, total), BG_COL)
    draw = ImageDraw.Draw(image)

    coord_font = _font(max(8, CELL_PX // 9))
    hazard_font = _font(max(9, CELL_PX // 8))
    title_font = _font(22)

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

    for wall_index in range(GRID + 1):
        for col in range(GRID):
            if h_walls[wall_index, col]:
                x0 = col * CELL_PX
                y0 = wall_index * CELL_PX
                x1 = x0 + CELL_PX + wall_px
                y1 = y0 + wall_px
                draw.rectangle([x0, y0, x1, y1], fill=WALL_COL)

    for row in range(GRID):
        for wall_index in range(GRID + 1):
            if v_walls[row, wall_index]:
                x0 = wall_index * CELL_PX
                y0 = row * CELL_PX
                x1 = x0 + wall_px
                y1 = y0 + CELL_PX + wall_px
                draw.rectangle([x0, y0, x1, y1], fill=WALL_COL)

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
                font=coord_font,
            )

            if (row, col) == start:
                label = "START"
            elif (row, col) == goal:
                label = "GOAL"
            elif (row, col) in hazards:
                label = HAZARD_SHORT[hazards[(row, col)]]
            else:
                label = None

            if label is not None:
                bbox = draw.textbbox((0, 0), label, font=hazard_font)
                tw = bbox[2] - bbox[0]
                draw.text(
                    (cx - tw // 2, cy + CELL_PX // 8),
                    label,
                    fill=HAZ_COL,
                    font=hazard_font,
                )

    if title:
        draw.text((10, 10), title, fill=(0, 0, 0), font=title_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"Saved → {out_path}")


def render_legend(hazards, out_path):
    entries = [
        ("START", START_FILL, "Entry cell"),
        ("GOAL", GOAL_FILL, "Exit cell"),
    ] + [
        (HAZARD_SHORT[hazard], HAZARD_FILL.get(hazard, (200, 200, 200)), HAZARD_LABELS[hazard])
        for hazard in sorted({value for value in hazards.values()}, key=lambda item: item.value)
    ]

    row_height = 40
    swatch_width = 30
    width = 340
    height = row_height * len(entries) + 24
    image = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    font = _font(15)

    draw.text((8, 4), "LEGEND", fill=(0, 0, 0), font=_font(16))

    for index, (short, fill, label) in enumerate(entries):
        y = 24 + index * row_height
        draw.rectangle([8, y, 8 + swatch_width, y + swatch_width], fill=fill, outline=(80, 80, 80))
        draw.text(
            (8 + swatch_width + 8, y + 6),
            f"{short}  —  {label}",
            fill=(20, 20, 20),
            font=font,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"Saved → {out_path}")


def render_fire_turns(h_walls, v_walls, start, goal, hazards, out_dir, num_turns=4):
    current_hazards = dict(hazards)
    pivots = None

    for turn in range(num_turns + 1):
        out_path = out_dir / f"maze_turn_{turn}.png"
        render_annotated(
            h_walls,
            v_walls,
            start,
            goal,
            current_hazards,
            out_path,
            title=f"TURN {turn}",
        )

        if turn < num_turns:
            current_hazards, pivots = update_fire_in_hazards(current_hazards, pivots)


def print_summary(start, goal, hazards, maze_name):
    print(f"Maze        : {maze_name}")
    print(f"Start       : {start}")
    print(f"Goal        : {goal}")
    print(f"Hazard cells: {len(hazards)}")
    for hazard in sorted({value for value in hazards.values()}, key=lambda item: item.value):
        count = sum(1 for value in hazards.values() if value == hazard)
        print(f"  {HAZARD_LABELS[hazard]:<10} {count}")


def parse_args():
    parser = argparse.ArgumentParser(description="Render annotated maze outputs for a given maze asset pair.")
    parser.add_argument("--walls", default=DEFAULT_WALLS_PATH, help="Wall image path, relative to the project root by default.")
    parser.add_argument("--hazards", default=DEFAULT_HAZARDS_PATH, help="Hazard image path, relative to the project root by default.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to results/printer/<maze-name>.")
    parser.add_argument("--fire-turns", type=int, default=8, help="How many rotating-fire turn snapshots to render.")
    parser.add_argument("--no-fire-turns", action="store_true", help="Skip rendering the turn-by-turn fire snapshots.")
    return parser.parse_args()


def main():
    args = parse_args()
    walls_path = resolve_project_path(args.walls)
    hazards_path = resolve_project_path(args.hazards)
    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else default_output_dir(walls_path, hazards_path)
    )

    print("Loading maze...")
    _, h_walls, v_walls = load_maze(str(walls_path))
    start, goal = find_start_goal(h_walls)

    print("Loading hazards...")
    hazards = load_hazards(str(hazards_path))
    maze_name = maze_label_from_paths(walls_path, hazards_path)

    print_summary(start, goal, hazards, maze_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    render_annotated(
        h_walls,
        v_walls,
        start,
        goal,
        hazards,
        out_path=output_dir / "annotated_maze.png",
        title=maze_name,
    )
    render_legend(
        hazards,
        out_path=output_dir / "annotated_maze_legend.png",
    )

    if not args.no_fire_turns:
        print_fire_turns(hazards, num_turns=args.fire_turns)
        render_fire_turns(
            h_walls,
            v_walls,
            start,
            goal,
            hazards,
            out_dir=output_dir,
            num_turns=args.fire_turns,
        )

    print(f"\nPrinter output directory: {output_dir}")


if __name__ == "__main__":
    main()
