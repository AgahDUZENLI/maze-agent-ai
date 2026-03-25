from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from maze_reader import (
    GRID,
    load_maze,
    load_hazards,
    get_start,
    get_goal,
    can_move,
    get_hazard,
    Hazard,
)

# ---------------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------------
CELL_PX = 20
PATH_COLOR = (255, 0, 0)
START_COLOR = (0, 200, 0)
GOAL_COLOR = (200, 0, 0)
EMPTY_COLOR = (240, 240, 240)
WALL_COLOR = (0, 0, 0)


def render_agent_path(h_walls, v_walls, path, start, goal, out_path="results/agent_path.png"):
    size = GRID * CELL_PX
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 10)
    except Exception:
        font = ImageFont.load_default()

    # draw cells
    for row in range(GRID):
        for col in range(GRID):
            x0 = col * CELL_PX
            y0 = row * CELL_PX
            x1 = x0 + CELL_PX
            y1 = y0 + CELL_PX

            if (row, col) == start:
                fill = START_COLOR
            elif (row, col) == goal:
                fill = GOAL_COLOR
            else:
                fill = EMPTY_COLOR

            draw.rectangle([x0, y0, x1, y1], fill=fill)

    # draw horizontal walls
    for wi in range(GRID + 1):
        for col in range(GRID):
            if h_walls[wi, col]:
                draw.rectangle(
                    [
                        col * CELL_PX,
                        wi * CELL_PX,
                        col * CELL_PX + CELL_PX,
                        wi * CELL_PX + 2,
                    ],
                    fill=WALL_COLOR,
                )

    # draw vertical walls
    for row in range(GRID):
        for wi in range(GRID + 1):
            if v_walls[row, wi]:
                draw.rectangle(
                    [
                        wi * CELL_PX,
                        row * CELL_PX,
                        wi * CELL_PX + 2,
                        row * CELL_PX + CELL_PX,
                    ],
                    fill=WALL_COLOR,
                )

    def center(cell):
        r, c = cell
        return (c * CELL_PX + CELL_PX // 2, r * CELL_PX + CELL_PX // 2)

    # draw path lines
    for i in range(len(path) - 1):
        draw.line([center(path[i]), center(path[i + 1])], fill=PATH_COLOR, width=3)

    # draw step numbers
    for i, (r, c) in enumerate(path):
        x = c * CELL_PX + 2
        y = r * CELL_PX + 2
        draw.text((x, y), str(i), fill=(0, 0, 0), font=font)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Saved PNG → {out_path}")


DIRECTIONS = {
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


class World:
    def __init__(self, h_walls, v_walls, hazards):
        self.h_walls = h_walls
        self.v_walls = v_walls
        self.hazards = hazards

    def ask_wall(self, row, col, direction):
        return can_move(row, col, direction, self.h_walls, self.v_walls)

    def ask_hazard(self, row, col):
        return get_hazard(row, col, self.hazards)


class BlindAgent:
    def __init__(self, world, start, goal):
        self.world = world
        self.start = start
        self.goal = goal
        self.position = start

        self.visited = set()
        self.path = [start]
        self.wall_hits = 0
        self.steps = 0
        self.deaths = 0
        self.teleports = 0

        self.known_open = defaultdict(set)
        self.known_blocked = defaultdict(set)
        self.known_hazards = {}

    def try_move(self, direction):
        row, col = self.position
        dr, dc = DIRECTIONS[direction]
        next_cell = (row + dr, col + dc)

        allowed = self.world.ask_wall(row, col, direction)

        if not allowed:
            self.known_blocked[(row, col)].add(direction)
            self.wall_hits += 1
            return False

        self.position = next_cell
        self.path.append(next_cell)
        self.steps += 1

        self.known_open[(row, col)].add(next_cell)
        self.known_open[next_cell].add((row, col))

        hazard = self.world.ask_hazard(*next_cell)
        if hazard is not None:
            self.known_hazards[next_cell] = hazard

        return True

    def handle_current_hazard(self):
        hazard = self.known_hazards.get(self.position)

        if hazard == Hazard.FIRE:
            print(f"FIRE at {self.position} -> respawn")
            self.deaths += 1
            self.position = self.start
            self.path.append(self.start)

        elif hazard == Hazard.CONFUSION:
            print(f"CONFUSION at {self.position}")

        elif hazard in {Hazard.TP_GREEN, Hazard.TP_YELLOW, Hazard.TP_PURPLE}:
            print(f"TELEPORT at {self.position}")
            matches = [
                cell for cell, hz in self.world.hazards.items()
                if hz == hazard and cell != self.position
            ]
            if matches:
                self.teleports += 1
                destination = matches[0]
                self.position = destination
                self.path.append(destination)

    def dfs(self, cell):
        self.position = cell
        self.visited.add(cell)

        if cell == self.goal:
            return True

        for direction in ["up", "right", "down", "left"]:
            row, col = cell
            dr, dc = DIRECTIONS[direction]
            next_cell = (row + dr, col + dc)

            if next_cell in self.visited:
                continue

            self.position = cell
            moved = self.try_move(direction)

            if not moved:
                continue

            self.handle_current_hazard()
            current = self.position

            if current == self.goal:
                self.visited.add(current)
                return True

            if current not in self.visited:
                found = self.dfs(current)
                if found:
                    return True

            if current == next_cell:
                back_dir = OPPOSITE[direction]
                self.try_move(back_dir)
                self.position = cell

        return False

    def solve(self):
        print(f"Start: {self.start}")
        print(f"Goal : {self.goal}")

        found = self.dfs(self.start)

        print("\n=== BLIND AGENT SUMMARY ===")
        print(f"Found goal    : {found}")
        print(f"Wall hits     : {self.wall_hits}")
        print(f"Deaths        : {self.deaths}")
        print(f"Teleports     : {self.teleports}")
        print(f"Steps moved   : {self.steps}")
        print(f"Visited cells : {len(self.visited)}")
        print(f"Known hazards : {len(self.known_hazards)}")

        print(f"\nPath length: {len(self.path)}")
        return found


if __name__ == "__main__":
    image, h_walls, v_walls = load_maze("MAZE_0.png")
    hazards = load_hazards("MAZE_1.png")

    start = get_start(h_walls)
    goal = get_goal(h_walls)

    world = World(h_walls, v_walls, hazards)
    agent = BlindAgent(world, start, goal)
    agent.solve()

    render_agent_path(
        h_walls,
        v_walls,
        agent.path,
        start,
        goal,
        out_path="results/agent_path.png",
    )