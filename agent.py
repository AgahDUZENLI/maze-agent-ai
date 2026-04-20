import random
import numpy as np
from collections import deque
from heapq import heappush, heappop
from typing import List, Tuple, Optional, Set, Dict

ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3
ACTIONS      = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

INVERT = {0: 1, 1: 0, 2: 3, 3: 2}

DELTAS = {
    ACTION_UP:    (-1,  0),
    ACTION_DOWN:  ( 1,  0),
    ACTION_LEFT:  ( 0, -1),
    ACTION_RIGHT: ( 0,  1),
}

GRID = 64


class HybridAgent:

    def __init__(self):

        # ── Permanent map (never clears) ──────────────────────────────────────
        self.walls:     Set[Tuple] = set()   # (row, col, action)
        self.teleports: Dict[Tuple, Tuple] = {}
        self.confuse:   Set[Tuple] = set()
        self.dead_cells: Set[Tuple] = set()  # clears each episode (fire moves)

        # ── Visited tracking ──────────────────────────────────────────────────
        self.visited:         Set[Tuple] = set()  # global — for A* and metrics
        self.episode_visited: Set[Tuple] = set()  # per-episode — for BFS frontier

        # ── Q-table ───────────────────────────────────────────────────────────
        self.q_table = np.zeros((GRID, GRID, 4), dtype=np.float32)
        self.alpha   = 0.2
        self.gamma   = 0.95

        # ── State ─────────────────────────────────────────────────────────────
        self.phase        = 1
        self.current_pos: Optional[Tuple] = None
        self.start_pos:   Optional[Tuple] = None
        self.goal_pos:    Optional[Tuple] = None
        self.prev_pos:    Optional[Tuple] = None
        self.prev_action: Optional[int]   = None
        self.action_queue: List[int]      = []
        self.is_confused  = False

        # ── Metrics ───────────────────────────────────────────────────────────
        self.total_episodes      = 0
        self.successful_episodes = 0
        self.total_turns_ever    = 0
        self.total_deaths_ever   = 0
        self.all_path_lengths:   List[int] = []
        self.all_turns:          List[int] = []
        self.all_deaths:         List[int] = []
        self.episode_turns    = 0
        self.episode_deaths   = 0
        self.episode_confused = 0
        self.episode_cells:   List[Tuple] = []
        self.optimal_path:    List[int]   = []

        self.last_planned_actions: List[int] = []
        self.last_submitted_actions: List[int] = []

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────

    def reset_episode(self, start_pos: Tuple) -> None:
        self.last_planned_actions = []
        self.last_submitted_actions = []
        self.start_pos        = start_pos
        self.current_pos      = start_pos
        self.prev_pos         = None
        self.prev_action      = None
        self.action_queue     = []
        self.is_confused      = False
        self.dead_cells       = set()          # fire rotates each episode
        self.episode_visited  = {start_pos}   # BFS frontier resets each episode
        self.visited.add(start_pos)
        self.episode_turns    = 0
        self.episode_deaths   = 0
        self.episode_confused = 0
        self.episode_cells    = [start_pos]
        self.total_episodes  += 1

        if self.phase == 2 and self.goal_pos is not None:
            self.action_queue = self._astar(self.start_pos, self.goal_pos)
            self.optimal_path = list(self.action_queue)

    def _trusted_prefix(self, path: List[int], max_len: int = 5) -> List[int]:
        """
        Return the longest safe prefix (up to 5 actions) that we are willing
        to batch in one turn.

        We only batch through plain known cells:
        - no known fire
        - no known confusion cells
        - no teleport pads
        - no blocked edges
        """
        if not path or self.current_pos is None:
            return []

        # keep batching conservative while confused
        if self.is_confused:
            return []

        r, c = self.current_pos
        prefix = []

        for a in path[:max_len]:
            if not self._can_move(r, c, a, ignore_fire=False):
                break

            dr, dc = DELTAS[a]
            nr, nc = r + dr, c + dc

            if (nr, nc) in self.dead_cells:
                break
            if (nr, nc) in self.confuse:
                break
            if (nr, nc) in self.teleports:
                break

            prefix.append(a)
            r, c = nr, nc

        return prefix

    def _remember_position(self, pos: Tuple[int, int]) -> None:
        self.visited.add(pos)
        self.episode_visited.add(pos)
        self.episode_cells.append(pos)
        self.current_pos = pos
    # ─────────────────────────────────────────────────────────────────────────
    # MAP
    # ─────────────────────────────────────────────────────────────────────────

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < GRID and 0 <= c < GRID

    def _can_move(self, r: int, c: int, a: int, ignore_fire=False) -> bool:
        if (r, c, a) in self.walls:
            return False
        dr, dc = DELTAS[a]
        nr, nc = r + dr, c + dc
        if not self._in_bounds(nr, nc):
            return False
        if not ignore_fire and (nr, nc) in self.dead_cells:
            return False
        return True

    def _neighbors(self, r: int, c: int, ignore_fire=False) -> List[Tuple]:
        out = []
        for a in ACTIONS:
            if self._can_move(r, c, a, ignore_fire):
                dr, dc = DELTAS[a]
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.teleports:
                    nr, nc = self.teleports[(nr, nc)]
                out.append((nr, nc, a))
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 — Biased BFS toward top of maze
    # ─────────────────────────────────────────────────────────────────────────

    def _bfs_explore(self, ignore_fire=False) -> List[int]:
        """
        BFS to nearest unvisited cell THIS episode.

        The priority queue is sorted by (row, distance) so cells closer
        to row 0 (the top, where the goal lives) are preferred when two
        frontiers are equally distant. This makes the agent naturally
        drift upward toward the exit instead of wandering randomly.
        """
        if self.current_pos is None:
            return []

        # heap: (priority, distance, position, path)
        # priority = row of destination (lower row = closer to top = better)
        heap = [(self.current_pos[0], 0, self.current_pos, [])]
        best: Dict[Tuple, int] = {self.current_pos: 0}

        while heap:
            _, dist, (r, c), path = heappop(heap)

            for nr, nc, action in self._neighbors(r, c, ignore_fire):
                nd = dist + 1
                if best.get((nr, nc), 999999) <= nd:
                    continue
                best[(nr, nc)] = nd
                new_path = path + [action]

                if (nr, nc) not in self.episode_visited:
                    return new_path   # found nearest unvisited, biased upward

                heappush(heap, (nr, nd, (nr, nc), new_path))

        return []

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — A*
    # ─────────────────────────────────────────────────────────────────────────

    def _astar(self, start: Tuple, goal: Tuple, ignore_fire=False) -> List[int]:
        if not start or not goal:
            return []

        def h(r, c):
            return abs(r - goal[0]) + abs(c - goal[1])

        heap   = [(h(*start), 0, start, [])]
        best_g: Dict[Tuple, int] = {}

        while heap:
            _, g, (r, c), path = heappop(heap)

            if (r, c) == goal:
                return path

            if best_g.get((r, c), 999999) <= g:
                continue
            best_g[(r, c)] = g

            for nr, nc, a in self._neighbors(r, c, ignore_fire):
                ng = g + 1
                if best_g.get((nr, nc), 999999) <= ng:
                    continue
                heappush(heap, (ng + h(nr, nc), ng, (nr, nc), path + [a]))

        return []

    # ─────────────────────────────────────────────────────────────────────────
    # Q-TABLE
    # ─────────────────────────────────────────────────────────────────────────

    def _update_q(self, prev, action, reward, new):
        r, c   = prev
        nr, nc = new
        old    = self.q_table[r, c, action]
        best_n = float(np.max(self.q_table[nr, nc]))
        self.q_table[r, c, action] = old + self.alpha * (reward + self.gamma * best_n - old)

    # ─────────────────────────────────────────────────────────────────────────
    # PROCESS RESULT
    # ─────────────────────────────────────────────────────────────────────────

    def _process_result(self, result) -> None:
        new_pos = result.current_position

        batched_turn = len(self.last_planned_actions) > 1

        # If we batched actions, only keep facts that are actually safe to infer.
        # Do NOT try to localize which exact intermediate action caused the event.
        if batched_turn:
            if result.is_dead:
                self.dead_cells.add(new_pos)
                self.episode_deaths += 1
                self.total_deaths_ever += 1
                self.current_pos = self.start_pos
                self.action_queue = []
                return

            if result.is_goal_reached:
                self.goal_pos = new_pos
                self._remember_position(new_pos)
                self._finish_episode(success=True)
                self.action_queue = []

                if self.phase == 1:
                    self.phase = 2
                    self.optimal_path = self._astar(self.start_pos, self.goal_pos)

                return

            if result.is_confused:
                self.is_confused = not self.is_confused
                self.episode_confused += 1
                self.action_queue = []

            # final position is trustworthy, intermediate causes are not
            self._remember_position(new_pos)

            # if something unexpected happened in a batched turn, drop back to cautious mode
            if result.wall_hits > 0 or result.teleported or result.is_confused:
                self.action_queue = []

            return
        
        # ── WALL ──────────────────────────────────────────────────────────────
        if result.wall_hits > 0 and self.prev_pos and self.prev_action is not None:
            self.walls.add((*self.prev_pos, self.prev_action))
            dr, dc = DELTAS[self.prev_action]
            nr, nc = self.prev_pos[0] + dr, self.prev_pos[1] + dc
            if self._in_bounds(nr, nc):
                self.walls.add((nr, nc, INVERT[self.prev_action]))
            if self.phase == 1:
                self._update_q(self.prev_pos, self.prev_action, -2, self.prev_pos)
            self.action_queue = []
            return  # didn't move

        # ── DEATH ─────────────────────────────────────────────────────────────
        if result.is_dead:
            self.dead_cells.add(new_pos)
            if self.phase == 1 and self.prev_pos and self.prev_action is not None:
                self._update_q(self.prev_pos, self.prev_action, -100, new_pos)
                self.q_table[new_pos[0], new_pos[1], :] = -200.0
            self.episode_deaths    += 1
            self.total_deaths_ever += 1
            self.current_pos        = self.start_pos
            self.action_queue       = []
            if self.phase == 2 and self.goal_pos:
                self.action_queue = self._astar(self.start_pos, self.goal_pos)
            return

        # ── TELEPORT ──────────────────────────────────────────────────────────
        if result.teleported and self.prev_pos and self.prev_action is not None:
            dr, dc   = DELTAS[self.prev_action]
            pad      = (self.prev_pos[0] + dr, self.prev_pos[1] + dc)
            if pad != new_pos:
                self.teleports[pad]     = new_pos
                self.teleports[new_pos] = pad
            self.action_queue = []
            if self.phase == 2 and self.goal_pos:
                self.action_queue = self._astar(new_pos, self.goal_pos)

        # ── CONFUSION ─────────────────────────────────────────────────────────
        if result.is_confused:
            self.is_confused = not self.is_confused
            self.confuse.add(new_pos)
            self.episode_confused += 1
            if self.phase == 1 and self.prev_pos and self.prev_action is not None:
                self._update_q(self.prev_pos, self.prev_action, -5, new_pos)
            self.action_queue = []

        # ── GOAL ──────────────────────────────────────────────────────────────
        if result.is_goal_reached:
            self.goal_pos = new_pos
            if self.phase == 1 and self.prev_pos and self.prev_action is not None:
                self._update_q(self.prev_pos, self.prev_action, +500, new_pos)
            self.visited.add(new_pos)
            self.episode_visited.add(new_pos)
            self.episode_cells.append(new_pos)
            self.current_pos = new_pos
            self._finish_episode(success=True)
            self.action_queue = []
            if self.phase == 1:
                self.phase        = 2
                self.optimal_path = self._astar(self.start_pos, self.goal_pos)
                steps = len(self.optimal_path)
                print(f"\n  ★ Goal found at {new_pos}! Switching to A* SPEED RUN.")
                print(f"  ★ Optimal path : {steps} steps ({steps//5+1} turns)")
            return

        # ── NORMAL MOVE ───────────────────────────────────────────────────────
        is_new = new_pos not in self.visited
        if self.phase == 1 and self.prev_pos and self.prev_action is not None:
            self._update_q(self.prev_pos, self.prev_action, +10 if is_new else -1, new_pos)
        self.visited.add(new_pos)
        self.episode_visited.add(new_pos)
        self.episode_cells.append(new_pos)
        self.current_pos = new_pos

    # ─────────────────────────────────────────────────────────────────────────
    # PLAN TURN
    # ─────────────────────────────────────────────────────────────────────────

    def plan_turn(self, last_result=None) -> List[int]:
        if last_result is not None:
            self._process_result(last_result)

        self.episode_turns += 1
        self.total_turns_ever += 1

        # Phase 2: follow known path, batch only trusted prefixes
        if self.phase == 2:
            if not self.action_queue and self.goal_pos and self.current_pos:
                self.action_queue = self._astar(self.current_pos, self.goal_pos)

            if self.action_queue:
                trusted = self._trusted_prefix(self.action_queue, max_len=5)

                if trusted:
                    self.action_queue = self.action_queue[len(trusted):]
                    return self._submit(trusted)

                # If the path is not trusted enough to batch, take only one step.
                next_step = self.action_queue.pop(0)
                return self._submit([next_step])

            return self._submit([random.choice(ACTIONS)])

        # Phase 1: cautious exploration = 1 action per turn
        if not self.action_queue:
            self.action_queue = self._bfs_explore()

        if self.action_queue:
            return self._submit([self.action_queue.pop(0)])

        return self._submit([random.choice(ACTIONS)])

    def _submit(self, desired_actions: List[int]) -> List[int]:
        desired_actions = desired_actions or [random.choice(ACTIONS)]

        if self.current_pos is not None:
            self.prev_pos = self.current_pos
            self.prev_action = desired_actions[0]   # world-direction action

        self.last_planned_actions = list(desired_actions)

        # If confused, submit the inverse so the environment's inversion
        # produces the world move we actually want.
        submitted = [
            INVERT[a] if self.is_confused else a
            for a in desired_actions
        ]

        self.last_submitted_actions = list(submitted)
        return submitted

    # ─────────────────────────────────────────────────────────────────────────
    # BOOKKEEPING
    # ─────────────────────────────────────────────────────────────────────────

    def _finish_episode(self, success: bool) -> None:
        if success:
            self.successful_episodes += 1
            self.all_path_lengths.append(len(self.episode_cells))
            self.all_turns.append(self.episode_turns)
        self.all_deaths.append(self.episode_deaths)

    def finish_episode_timeout(self) -> None:
        self._finish_episode(success=False)

    # ─────────────────────────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        t, s = self.total_episodes, self.successful_episodes
        return {
            "phase":            self.phase,
            "total_episodes":   t,
            "successful":       s,
            "success_rate":     round(s / t * 100, 1) if t else 0.0,
            "avg_path_length":  round(float(np.mean(self.all_path_lengths)), 1) if self.all_path_lengths else 0.0,
            "avg_turns":        round(float(np.mean(self.all_turns)), 1) if self.all_turns else 0.0,
            "death_rate":       round(self.total_deaths_ever / max(self.total_turns_ever, 1), 4),
            "map_completeness": round(len(self.visited) / (GRID * GRID), 4),
            "goal_found":       self.goal_pos is not None,
            "goal_pos":         self.goal_pos,
            "optimal_path_len": len(self.optimal_path),
            "unique_cells":     len(self.visited),
            "walls_mapped":     len(self.walls),
            "deaths_mapped":    len(self.dead_cells),
            "teleports_mapped": len(self.teleports) // 2,
            "confuse_mapped":   len(self.confuse),
        }

    def print_metrics(self) -> None:
        m = self.get_metrics()
        print("\n" + "=" * 55)
        print("AGENT PERFORMANCE METRICS")
        print("=" * 55)
        print(f"  Phase              : {'SPEED RUN (A*)' if m['phase']==2 else 'EXPLORING (Biased BFS)'}")
        print(f"  Optimal path       : {m['optimal_path_len']} steps")
        print(f"  Episodes run       : {m['total_episodes']}")
        print(f"  Successful         : {m['successful']}")
        print(f"  Success rate       : {m['success_rate']}%")
        print(f"  Avg path length    : {m['avg_path_length']} cells")
        print(f"  Avg turns to solve : {m['avg_turns']} turns")
        print(f"  Death rate         : {m['death_rate']}")
        print("─" * 55)
        print(f"  Map completeness   : {m['map_completeness']*100:.1f}%")
        print(f"  Goal               : {m['goal_pos']}")
        print(f"  Walls mapped       : {m['walls_mapped']}")
        print(f"  Teleports mapped   : {m['teleports_mapped']} pairs")
        print(f"  Confusion cells    : {m['confuse_mapped']}")
        print("=" * 55)