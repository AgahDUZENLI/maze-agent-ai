import os
from PIL import Image, ImageDraw

from maze_reader import GRID, Hazard, update_fire_in_hazards


CELL_SIZE = 10


class MazeVisualizer:
    def __init__(self, maze_path, gif_fps=10, gif_skip=1):
        self.base_img = self._build_base_image(maze_path)
        self.frames = []
        self.gif_skip = gif_skip
        self.frame_duration = int(1000 / gif_fps)

        # 🔥 persistent memory
        self.discovered_cells = set()      # explored path
        self.discovered_hazards = set()   # ALL hazards ever seen

    # ───────────────────────────────────────────────
    def _build_base_image(self, maze_path):
        img = Image.open(maze_path).convert("RGBA")
        img = img.resize((GRID * CELL_SIZE, GRID * CELL_SIZE))
        return img

    # ───────────────────────────────────────────────
    def _generate_fire_phases(self, hazards, fire_groups):
        phases = []

        cur_hazards = dict(hazards)
        cur_groups = list(fire_groups)

        for _ in range(4):
            phases.append((dict(cur_hazards), cur_groups))
            cur_hazards, cur_groups = update_fire_in_hazards(
                cur_hazards, cur_groups
            )

        return phases

    # ───────────────────────────────────────────────
    def _render_phase_frame(self, env, hazards, fire_groups, path, deaths):

        frame = self.base_img.copy()
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        # 🗺️ explored cells (subtle)
        for r, c in self.discovered_cells:
            x0 = c * CELL_SIZE
            y0 = r * CELL_SIZE
            x1 = x0 + CELL_SIZE
            y1 = y0 + CELL_SIZE

            draw_overlay.rectangle(
                [x0, y0, x1, y1],
                fill=(200, 200, 200, 50)
            )

        frame = Image.alpha_composite(frame, overlay)
        draw = ImageDraw.Draw(frame)

        # 🔥 FIRE (persist once seen)
        for group, pivot in fire_groups:
            pr, pc = pivot

            for r, c in group:
                x0 = c * CELL_SIZE
                y0 = r * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE

                seen = (r, c) in self.discovered_hazards
                color = (255, 80, 0) if seen else (255, 180, 150)

                draw.rectangle([x0, y0, x1, y1], fill=color)

            # pivot
            px = pc * CELL_SIZE + CELL_SIZE // 2
            py = pr * CELL_SIZE + CELL_SIZE // 2

            if (pr, pc) in self.discovered_hazards:
                draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(0, 0, 0))
            else:
                draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(180, 180, 180))

        # 🧠 OTHER HAZARDS
        for (r, c), hz in hazards.items():
            if hz == Hazard.FIRE:
                continue

            x0 = c * CELL_SIZE
            y0 = r * CELL_SIZE
            x1 = x0 + CELL_SIZE
            y1 = y0 + CELL_SIZE

            seen = (r, c) in self.discovered_hazards

            if hz == Hazard.CONFUSION:
                color = (160, 120, 60) if seen else (210, 190, 150)

            elif "tp" in hz.value:
                color = (100, 200, 255) if seen else (180, 220, 240)

            else:
                continue

            draw.rectangle([x0, y0, x1, y1], fill=color)

        # 🟦 PATH
        if len(path) > 1:
            pts = [
                (c * CELL_SIZE + CELL_SIZE // 2,
                 r * CELL_SIZE + CELL_SIZE // 2)
                for r, c in path
            ]
            draw.line(pts, fill=(0, 0, 255), width=2)

        # 🟥 DEATHS
        for r, c in deaths:
            x0 = c * CELL_SIZE
            y0 = r * CELL_SIZE
            x1 = x0 + CELL_SIZE
            y1 = y0 + CELL_SIZE
            draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0))

        # 🟩 AGENT
        r, c = env.agent_pos
        x0 = c * CELL_SIZE
        y0 = r * CELL_SIZE
        x1 = x0 + CELL_SIZE
        y1 = y0 + CELL_SIZE
        draw.rectangle([x0, y0, x1, y1], fill=(0, 255, 0))

        return frame.convert("RGB")

    # ───────────────────────────────────────────────
    def capture_frame(self, agent, env, path_so_far, deaths_so_far, episode_num):

        if env.turn % self.gif_skip != 0:
            return

        # ✅ track explored cells
        self.discovered_cells.add(env.agent_pos)
        self.discovered_cells.update(path_so_far)
        self.discovered_cells.update(deaths_so_far)

        # 🔥 FIX: track ALL hazards seen this turn
        for cell, hz in env.hazards.items():
            if hz is not None:
                self.discovered_hazards.add(cell)

        phases = self._generate_fire_phases(env.hazards, env.fire_pivots)

        for hz, groups in phases:
            frame = self._render_phase_frame(
                env, hz, groups, path_so_far, deaths_so_far
            )
            self.frames.append(frame)

    # ───────────────────────────────────────────────
    def save_episode(self, episode_num, agent, env,
                     path_taken, death_cells, output_dir):

        os.makedirs(output_dir, exist_ok=True)

        gif_path = os.path.join(output_dir, f"episode_{episode_num}.gif")
        png_path = os.path.join(output_dir, f"episode_{episode_num}.png")

        final_frame = self._render_phase_frame(
            env,
            env.hazards,
            env.fire_pivots,
            path_taken,
            death_cells
        )
        self.frames.append(final_frame)

        if self.frames:
            self.frames[0].save(
                gif_path,
                save_all=True,
                append_images=self.frames[1:],
                duration=self.frame_duration,
                loop=0
            )

        final_frame.save(png_path)

        self.frames = []

        return png_path, gif_path