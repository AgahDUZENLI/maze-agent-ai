"""
visualizer.py
=============
Draws the agent's episode onto the maze image and saves a JPEG.

Shows:
  - Maze walls (from original PNG)
  - Green cells  = visited this episode
  - Blue path    = the exact route the agent took
  - Red X        = where the agent died (fire)
  - Gold star    = goal location
  - Cyan dot     = start location
  - Orange cells = known teleporter cells
  - Purple cells = known confusion traps
  - Fire cells   = current fire positions (from environment)

Usage:
    from visualizer import MazeVisualizer
    viz = MazeVisualizer("MAZE_0.png")
    viz.save_episode(episode_num, agent, env, path_taken)
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Set, Dict, Optional

# Must match maze_reader.py exactly
GRID  = 64
WALL  = 2
STEP  = 16
INNER = 14


def cell_to_pixel(row: int, col: int) -> Tuple[int, int]:
    """Return top-left pixel (x, y) of a cell."""
    x = WALL + col * STEP
    y = WALL + row * STEP
    return x, y


def cell_center_px(row: int, col: int) -> Tuple[int, int]:
    """Return center pixel (x, y) of a cell."""
    x = WALL + col * STEP + STEP // 2
    y = WALL + row * STEP + STEP // 2
    return x, y


class MazeVisualizer:
    def __init__(self, maze_path: str):
        self.maze_path = maze_path
        # Load original maze image as base
        self.base_image = Image.open(maze_path).convert("RGB")

    def _fresh_canvas(self) -> Image.Image:
        """Return a fresh copy of the maze image to draw on."""
        return self.base_image.copy()

    def save_episode(
        self,
        episode_num:  int,
        agent,                    # HybridAgent instance
        env,                      # MazeEnvironment instance
        path_taken:   List[Tuple[int, int]],   # list of (row,col) positions this episode
        death_cells:  List[Tuple[int, int]],   # where agent died this episode
        output_dir:   str = ".",
    ):
        """
        Draw the episode and save as JPEG.

        Parameters
        ----------
        episode_num  : episode number (used in filename)
        agent        : the HybridAgent (for map knowledge)
        env          : the MazeEnvironment (for fire positions, goal, start)
        path_taken   : ordered list of (row,col) the agent actually visited
        death_cells  : list of (row,col) where agent died this episode
        output_dir   : folder to save the image
        """
        img  = self._fresh_canvas()
        draw = ImageDraw.Draw(img, "RGBA")

        cell_size = STEP   # pixels per cell (16)
        inner     = INNER  # inner cell size (14)

        # ── Helper: fill a cell with a color ──────────────────────────────────
        def fill_cell(row, col, color):
            x, y = cell_to_pixel(row, col)
            # Draw inside the cell, leaving wall pixels intact
            draw.rectangle(
                [x + 1, y + 1, x + inner, y + inner],
                fill=color
            )

        # ── 1. Visited cells (light green, semi-transparent) ──────────────────
        for (r, c) in agent.visited:
            fill_cell(r, c, (144, 238, 144, 80))   # light green, 80/255 alpha

        # ── 2. Current fire positions (red-orange) ────────────────────────────
        from maze_reader import Hazard
        for (r, c), hz in env.hazards.items():
            if hz == Hazard.FIRE:
                fill_cell(r, c, (255, 80, 0, 160))   # orange-red
            elif hz == Hazard.PUSH_UP:
                fill_cell(r, c, (70, 130, 180, 150))
            elif hz == Hazard.PUSH_LEFT:
                fill_cell(r, c, (100, 149, 237, 150))

        # ── 3. Known confusion cells (purple) ─────────────────────────────────
        for (r, c) in agent.confuse:
            fill_cell(r, c, (180, 0, 255, 140))   # purple

        # ── 4. Known teleporter cells (orange) ────────────────────────────────
        for (r, c), dest in agent.teleports.items():
            fill_cell(r, c, (255, 165, 0, 160))   # orange

        # ── 5. Path taken this episode (blue line) ────────────────────────────
        if len(path_taken) >= 2:
            # Draw line connecting each step
            pixel_path = [cell_center_px(r, c) for r, c in path_taken]
            for i in range(len(pixel_path) - 1):
                draw.line(
                    [pixel_path[i], pixel_path[i+1]],
                    fill=(30, 144, 255, 200),   # dodger blue
                    width=2
                )

        # ── 6. Death locations (red X) ────────────────────────────────────────
        for (r, c) in death_cells:
            cx, cy = cell_center_px(r, c)
            s = 4   # size of X arms
            draw.line([(cx-s, cy-s), (cx+s, cy+s)], fill=(220, 0, 0), width=2)
            draw.line([(cx+s, cy-s), (cx-s, cy+s)], fill=(220, 0, 0), width=2)

        # ── 7. Start cell (cyan dot) ───────────────────────────────────────────
        sr, sc = env.start
        cx, cy = cell_center_px(sr, sc)
        draw.ellipse([(cx-4, cy-4), (cx+4, cy+4)], fill=(0, 255, 255))

        # ── 8. Goal cell (gold star / filled circle) ───────────────────────────
        gr, gc = env.goal
        cx, cy = cell_center_px(gr, gc)
        draw.ellipse([(cx-5, cy-5), (cx+5, cy+5)], fill=(255, 215, 0))

        # ── 9. Current agent position (white dot) ─────────────────────────────
        if agent.current_pos:
            ar, ac = agent.current_pos
            cx, cy = cell_center_px(ar, ac)
            draw.ellipse([(cx-3, cy-3), (cx+3, cy+3)], fill=(255, 255, 255))

        # ── 10. Legend text ───────────────────────────────────────────────────
        # Draw a small legend in the top-left corner
        legend_x, legend_y = 4, 4
        legend_items = [
            ((144, 238, 144), "visited"),
            ((30, 144, 255),  "path"),
            ((255, 80, 0),    "fire"),
            ((220, 0, 0),     "death"),
            ((0, 255, 255),   "start"),
            ((255, 215, 0),   "goal"),
            ((180, 0, 255),   "confusion"),
            ((255, 165, 0),   "teleport"),
            ((70, 130, 180),  "push-up"),
            ((100, 149, 237), "push-left"),
        ]

        # Semi-transparent legend background
        legend_h = len(legend_items) * 10 + 6
        draw.rectangle(
            [legend_x, legend_y, legend_x + 72, legend_y + legend_h],
            fill=(0, 0, 0, 160)
        )

        for i, (color, label) in enumerate(legend_items):
            y = legend_y + 4 + i * 10
            # Color swatch
            draw.rectangle([legend_x+2, y, legend_x+8, y+7], fill=color)
            # Label (draw without font — just use default)
            draw.text((legend_x+11, y), label, fill=(255, 255, 255))

        # ── 11. Episode info header ────────────────────────────────────────────
        # Draw episode number and stats at bottom
        m           = agent.get_metrics()
        stats_text  = (
            f"Ep {episode_num} | "
            f"cells={len(agent.visited)} | "
            f"deaths={len(death_cells)} | "
            f"goal={'FOUND' if agent.goal_pos else 'not found'} | "
            f"success={m['success_rate']}%"
        )

        # Background strip at bottom
        img_w, img_h = img.size
        draw.rectangle(
            [0, img_h - 14, img_w, img_h],
            fill=(0, 0, 0, 200)
        )
        draw.text((4, img_h - 12), stats_text, fill=(255, 255, 255))

        # ── Save ──────────────────────────────────────────────────────────────
        import os
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"episode_{episode_num:03d}.jpg")
        img.convert("RGB").save(out_path, "JPEG", quality=92)
        print(f"  [VIZ] Saved → {out_path}")

        return out_path
