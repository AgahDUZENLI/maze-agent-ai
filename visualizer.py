"""
Legacy compatibility visualizer for the upstream repository layout.

This file keeps the old `visualizer.py` entry point available for static episode
snapshots. The official replay renderer remains `maze_visualizer.py`.
"""

from pathlib import Path

from PIL import Image, ImageDraw

from maze_reader import Hazard, update_fire_in_hazards

GRID = 64
WALL = 2
STEP = 16
INNER = 14


def cell_to_pixel(row, col):
    x = WALL + col * STEP
    y = WALL + row * STEP
    return x, y


def cell_center_px(row, col):
    x = WALL + col * STEP + STEP // 2
    y = WALL + row * STEP + STEP // 2
    return x, y


class MazeVisualizer:
    def __init__(self, maze_path):
        self.maze_path = maze_path
        self.base_image = Image.open(maze_path).convert("RGB")

    def _fresh_canvas(self):
        return self.base_image.copy()

    def _display_hazards(self, base_hazards, path_taken, events):
        hazards = dict(base_hazards)
        fire_pivots = None
        fire_rotations = 0

        if events:
            consumed_moves = 0
            for event in events:
                if not event.get("consumes_move", False):
                    continue
                consumed_moves += 1
                if consumed_moves % 5 == 0:
                    hazards, fire_pivots = update_fire_in_hazards(hazards, fire_pivots)
                    fire_rotations += 1
            return hazards, fire_rotations

        # Fall back to path length when no event log is available.
        for _ in range(max(0, len(path_taken) - 1) // 5):
            hazards, fire_pivots = update_fire_in_hazards(hazards, fire_pivots)
            fire_rotations += 1

        return hazards, fire_rotations

    def save_episode(
        self,
        episode_num,
        agent,
        env,
        path_taken,
        death_cells,
        events=None,
        rotate_fire=True,
        output_dir=".",
    ):
        image = self._fresh_canvas()
        draw = ImageDraw.Draw(image, "RGBA")

        def fill_cell(row, col, color):
            x, y = cell_to_pixel(row, col)
            draw.rectangle([x + 1, y + 1, x + INNER, y + INNER], fill=color)

        known_cells = getattr(agent, "knowledge", {}).get("cell_types", {})
        display_hazards = dict(env.hazards)
        fire_rotations = 0

        if rotate_fire:
            display_hazards, fire_rotations = self._display_hazards(
                env.hazards,
                path_taken,
                events,
            )

        for (row, col), cell_type in known_cells.items():
            if cell_type in {"start", "safe", "goal"}:
                fill_cell(row, col, (144, 238, 144, 80))
            elif cell_type == "confusion":
                fill_cell(row, col, (180, 0, 255, 140))
            elif cell_type == "teleport":
                fill_cell(row, col, (255, 165, 0, 160))
            elif cell_type == "lethal":
                fill_cell(row, col, (255, 110, 110, 120))

        for (row, col), hazard in display_hazards.items():
            if hazard == Hazard.FIRE:
                fill_cell(row, col, (255, 80, 0, 150))

        if len(path_taken) >= 2:
            pixel_path = [cell_center_px(row, col) for row, col in path_taken]
            for index in range(len(pixel_path) - 1):
                draw.line(
                    [pixel_path[index], pixel_path[index + 1]],
                    fill=(30, 144, 255, 210),
                    width=2,
                )

        for row, col in death_cells:
            cx, cy = cell_center_px(row, col)
            size = 4
            draw.line([(cx - size, cy - size), (cx + size, cy + size)], fill=(220, 0, 0), width=2)
            draw.line([(cx + size, cy - size), (cx - size, cy + size)], fill=(220, 0, 0), width=2)

        start_row, start_col = env.start
        goal_row, goal_col = env.goal
        start_cx, start_cy = cell_center_px(start_row, start_col)
        goal_cx, goal_cy = cell_center_px(goal_row, goal_col)
        draw.ellipse([(start_cx - 4, start_cy - 4), (start_cx + 4, start_cy + 4)], fill=(0, 255, 255))
        draw.ellipse([(goal_cx - 5, goal_cy - 5), (goal_cx + 5, goal_cy + 5)], fill=(255, 215, 0))

        current_row, current_col = path_taken[-1] if path_taken else env.start
        current_cx, current_cy = cell_center_px(current_row, current_col)
        draw.ellipse([(current_cx - 3, current_cy - 3), (current_cx + 3, current_cy + 3)], fill=(255, 255, 255))

        legend_x, legend_y = 4, 4
        legend_items = [
            ((144, 238, 144), "known"),
            ((30, 144, 255), "path"),
            ((255, 80, 0), "fire"),
            ((220, 0, 0), "death"),
            ((0, 255, 255), "start"),
            ((255, 215, 0), "goal"),
            ((180, 0, 255), "confusion"),
            ((255, 165, 0), "teleport"),
        ]
        legend_h = len(legend_items) * 10 + 6
        draw.rectangle([legend_x, legend_y, legend_x + 74, legend_y + legend_h], fill=(0, 0, 0, 160))
        for index, (color, label) in enumerate(legend_items):
            y = legend_y + 4 + index * 10
            draw.rectangle([legend_x + 2, y, legend_x + 8, y + 7], fill=color)
            draw.text((legend_x + 11, y), label, fill=(255, 255, 255))

        image_w, image_h = image.size
        footer = (
            f"Ep {episode_num} | cells={len(known_cells)} | "
            f"deaths={len(death_cells)} | fire turns={fire_rotations} | "
            f"goal={'FOUND' if known_cells.get(env.goal) == 'goal' else 'not found'}"
        )
        draw.rectangle([0, image_h - 14, image_w, image_h], fill=(0, 0, 0, 200))
        draw.text((4, image_h - 12), footer, fill=(255, 255, 255))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"episode_{episode_num:03d}.jpg"
        image.convert("RGB").save(out_path, "JPEG", quality=92)
        print(f"  [legacy-viz] Saved -> {out_path}")
        return str(out_path)


__all__ = ["MazeVisualizer"]
