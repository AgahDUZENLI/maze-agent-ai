import random
import numpy as np
from collections import deque
from heapq import heappush, heappop
from typing import List, Tuple, Optional, Set, Dict

ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3
ACTION_WAIT  = 4

MOVE_ACTIONS    = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
PLANNER_ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_WAIT]

INVERT = {0: 1, 1: 0, 2: 3, 3: 2}

OPPOSITE = {
    ACTION_UP: ACTION_DOWN,
    ACTION_DOWN: ACTION_UP,
    ACTION_LEFT: ACTION_RIGHT,
    ACTION_RIGHT: ACTION_LEFT,
}

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
        self.push_tiles: Dict[Tuple[int, int], int] = {}

        # Fire is not static. Learn dangerous cells by fire phase.
        self.dead_cells_by_phase: Dict[int, Set[Tuple[int, int]]] = {
            0: set(),
            1: set(),
            2: set(),
            3: set(),
        }

        # ── Visited tracking ──────────────────────────────────────────────────
        self.visited:         Set[Tuple] = set()  # global — for A* and metrics
        self.episode_visited: Set[Tuple] = set()  # per-episode — for BFS frontier
        self.episode_world_actions: List[int] = []
        self.safe_moves: Set[Tuple[int, int, int]] = set()

        # ── Q-table ───────────────────────────────────────────────────────────
        self.q_table = np.zeros((GRID, GRID, 4), dtype=np.float32)
        self.alpha   = 0.2
        self.gamma   = 0.95

        # ── Epsilon-greedy exploration schedule ───────────────────────────────
        # Starts fully exploratory; decays toward min after each episode so the
        # agent increasingly exploits what the Q-table has learned.
        self.epsilon       = 1.0    # current exploration rate
        self.epsilon_min   = 0.05   # never go fully greedy
        self.epsilon_decay = 0.995  # multiplied each episode

        # ── State ─────────────────────────────────────────────────────────────
        self.phase        = 1
        self.current_pos: Optional[Tuple] = None
        self.start_pos:   Optional[Tuple] = None
        self.goal_pos:    Optional[Tuple] = None
        self.prev_pos:    Optional[Tuple] = None
        self.prev_action: Optional[int]   = None
        self.action_queue: List[int]      = []
        self.is_confused  = False
        self.atomic_action_count = 0
        self.last_turn_start_tmod20 = 0

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
        self.atomic_action_count = 0
        self.last_turn_start_tmod20 = 0
        self.episode_visited  = {start_pos}   # BFS frontier resets each episode
        self.visited.add(start_pos)
        self.episode_turns    = 0
        self.episode_deaths   = 0
        self.episode_confused = 0
        self.episode_cells    = [start_pos]
        self.total_episodes  += 1
        self.episode_world_actions = []

        # Decay epsilon each episode: agent exploits Q-table more over time
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.phase == 2 and self.optimal_path:
            self.action_queue = list(self.optimal_path)

    def _record_safe_move(
        self,
        from_pos: Optional[Tuple[int, int]],
        action: Optional[int],
        to_pos: Tuple[int, int],
        teleported: bool = False
    ) -> None:
        if from_pos is None or action not in MOVE_ACTIONS:
            return

        fr, fc = from_pos
        tr, tc = to_pos

        # forward move is known safe because it just happened
        self.safe_moves.add((fr, fc, action))

        # if it was a normal adjacent move, reverse direction is also safe
        if not teleported and abs(fr - tr) + abs(fc - tc) == 1:
            self.safe_moves.add((tr, tc, OPPOSITE[action]))

    def _trusted_prefix(self, path: List[int], max_len: int = 5) -> List[int]:
        """
        Return the longest safe prefix (up to 5 actions) that we are willing
        to batch in one turn.

        This is time-aware: fire danger depends on the current time_mod_20.
        """
        if not path or self.current_pos is None:
            return []

        # keep batching conservative while confused
        if self.is_confused:
            return []

        r, c = self.current_pos
        tmod20 = self._current_tmod20()
        prefix = []

        for action in path[:max_len]:
            # Keep batching through WAIT allowed; it may be useful for timed corridors
            if action == ACTION_WAIT:
                prefix.append(action)
                tmod20 = (tmod20 + 1) % 20
                continue

            # avoid batching through known special cells
            dr, dc = DELTAS[action]
            pad = (r + dr, c + dc)

            nxt = self._transition(r, c, tmod20, action, ignore_fire=False)
            if nxt is None:
                break

            nr, nc, nt = nxt

            if pad in self.teleports:
                break
            if pad in self.push_tiles:
                break
            if (nr, nc) in self.confuse:
                break

            prefix.append(action)
            r, c, tmod20 = nr, nc, nt

        return prefix

    def _remember_position(self, pos: Tuple[int, int]) -> None:
        self.visited.add(pos)
        self.episode_visited.add(pos)
        self.episode_cells.append(pos)
        self.current_pos = pos

    def _current_tmod20(self) -> int:
        return self.atomic_action_count % 20

    def _phase_at(self, tmod20: int) -> int:
        return (tmod20 // 5) % 4

    def _advance_internal_clock(self, actions_executed: int) -> None:
        self.atomic_action_count += actions_executed

    def _mark_fire_death(self, cell: Tuple[int, int], tmod20: int) -> None:
        phase = self._phase_at(tmod20)
        self.dead_cells_by_phase[phase].add(cell)

    def _is_known_dead(self, cell: Tuple[int, int], tmod20: int) -> bool:
        phase = self._phase_at(tmod20)
        return cell in self.dead_cells_by_phase[phase]

    def _transition(self, r: int, c: int, tmod20: int, action: int,
                ignore_fire: bool = False, require_safe: bool = False):
        """
        Simulate one planner action in the agent's internal model.

        tmod20 means: number of atomic actions elapsed mod 20 BEFORE this action.
        Fire danger for a move is checked at the current phase, then time advances by 1.
        """
        next_t = (tmod20 + 1) % 20

        if action == ACTION_WAIT:
            return (r, c, next_t)
        
        if (r, c, action) in self.walls:
            return None

        if require_safe and action != ACTION_WAIT and (r, c, action) not in self.safe_moves:
            return None

        dr, dc = DELTAS[action]
        nr, nc = r + dr, c + dc

        if not self._in_bounds(nr, nc):
            return None

        # Maze-gamma directional push hazards resolve immediately on landing.
        # Assumption: each push tile forces at most one extra move inside the
        # same submitted action; it does not consume another atomic action.
        pad = (nr, nc)
        if pad in self.push_tiles:
            push_action = self.push_tiles[pad]
            if (nr, nc, push_action) not in self.walls:
                pdr, pdc = DELTAS[push_action]
                fr, fc = nr + pdr, nc + pdc
                if self._in_bounds(fr, fc):
                    nr, nc = fr, fc
            pad = (nr, nc)

        # teleport happens immediately on landing
        if pad in self.teleports:
            nr, nc = self.teleports[pad]

        # fire is checked using the phase at the START of the action
        if not ignore_fire and self._is_known_dead((nr, nc), tmod20):
            return None

        return (nr, nc, next_t)

    def _neighbors_time(self, r: int, c: int, tmod20: int,
                    ignore_fire: bool = False, require_safe: bool = False) -> List[Tuple[int, int, int, int]]:
        out = []
        for action in PLANNER_ACTIONS:
            nxt = self._transition(r, c, tmod20, action, ignore_fire, require_safe)
            if nxt is not None:
                nr, nc, nt = nxt
                out.append((nr, nc, nt, action))
        return out
    # ─────────────────────────────────────────────────────────────────────────
    # MAP
    # ─────────────────────────────────────────────────────────────────────────

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < GRID and 0 <= c < GRID

    def _can_move(self, r: int, c: int, a: int, ignore_fire=False) -> bool:
        if a == ACTION_WAIT:
            return True
        if (r, c, a) in self.walls:
            return False
        dr, dc = DELTAS[a]
        nr, nc = r + dr, c + dc
        if not self._in_bounds(nr, nc):
            return False
        if not ignore_fire and self._is_known_dead((nr, nc), self._current_tmod20()):
            return False
        return True

    def _neighbors(self, r: int, c: int, ignore_fire=False) -> List[Tuple]:
        out = []
        for a in MOVE_ACTIONS:
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
        Time-aware BFS to nearest unvisited cell THIS episode.

        State includes time_mod_20 so the agent can wait for fire to rotate
        if that produces a shorter safe route.

        The heap priority blends the cell's row position with its Q-value so
        that the search steers toward cells the Q-table considers promising
        (e.g. away from fire corridors, toward productive directions) while
        still guaranteeing it finds an unvisited cell.
        """
        if self.current_pos is None:
            return []

        start_t = self._current_tmod20()

        def q_priority(r, c):
            # Higher Q-value → lower heap number → explored sooner.
            # Scale factor 0.5 keeps Q-bias subordinate to row distance so
            # the upward bias is preserved but Q-learning meaningfully influences
            # which corridor the agent prefers when distance is equal.
            q_val = float(np.max(self.q_table[r, c]))
            return r - 0.5 * q_val

        # heap item: (priority, distance, (r, c), tmod20, path)
        r0, c0 = self.current_pos
        heap = [(q_priority(r0, c0), 0, self.current_pos, start_t, [])]
        best: Dict[Tuple[int, int, int], int] = {
            (r0, c0, start_t): 0
        }

        while heap:
            _, dist, (r, c), tmod20, path = heappop(heap)

            for nr, nc, nt, action in self._neighbors_time(r, c, tmod20, ignore_fire):
                nd = dist + 1
                state = (nr, nc, nt)

                if best.get(state, 999999) <= nd:
                    continue
                best[state] = nd

                new_path = path + [action]

                # Only count movement to a new cell as exploration progress;
                # don't "discover" a frontier by waiting in place.
                if action != ACTION_WAIT and (nr, nc) not in self.visited:
                    return new_path

                heappush(heap, (q_priority(nr, nc), nd, (nr, nc), nt, new_path))

        return []

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — A*
    # ─────────────────────────────────────────────────────────────────────────

    def _astar(
        self,
        start: Tuple,
        goal: Tuple,
        start_tmod20: Optional[int] = None,
        ignore_fire: bool = False,
        require_safe: bool = False
    ) -> List[int]:
        if not start or not goal:
            return []

        if start_tmod20 is None:
            start_t = self._current_tmod20()
        else:
            start_t = start_tmod20 % 20

        def h(r, c):
            return abs(r - goal[0]) + abs(c - goal[1])

        heap = [(h(*start), 0, start, start_t, [])]
        best_g: Dict[Tuple[int, int, int], int] = {}

        while heap:
            _, g, (r, c), tmod20, path = heappop(heap)

            if (r, c) == goal:
                return path

            state = (r, c, tmod20)
            if best_g.get(state, 999999) <= g:
                continue
            best_g[state] = g

            for nr, nc, nt, action in self._neighbors_time(r, c, tmod20, ignore_fire, require_safe):
                ng = g + 1
                next_state = (nr, nc, nt)

                if best_g.get(next_state, 999999) <= ng:
                    continue

                heappush(heap, (ng + h(nr, nc), ng, (nr, nc), nt, path + [action]))

        return []

    # ─────────────────────────────────────────────────────────────────────────
    # Q-TABLE
    # ─────────────────────────────────────────────────────────────────────────

    def _update_q(self, prev, action, reward, new):
        if action not in MOVE_ACTIONS:
            return

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
        start_tmod20 = self.last_turn_start_tmod20
        death_tmod20 = (start_tmod20 + max(result.actions_executed - 1, 0)) % 20

        batched_turn = len(self.last_planned_actions) > 1

        def save_successful_replay():
            overflow = len(self.last_planned_actions) - result.actions_executed
            successful_trace = list(self.episode_world_actions)

            if overflow > 0:
                successful_trace = successful_trace[:-overflow]

            # Try to compress the route using the currently known map,
            # starting from the reset state of a new episode (time = 0).
            compressed = []
            if self.start_pos is not None and self.goal_pos is not None:
                compressed = self._astar(
                    self.start_pos,
                    self.goal_pos,
                    start_tmod20=0,
                    require_safe=True
                )

            if compressed:
                if not self.optimal_path or len(compressed) < len(self.optimal_path):
                    self.optimal_path = compressed
            else:
                if not self.optimal_path or len(successful_trace) < len(self.optimal_path):
                    self.optimal_path = successful_trace

        # If we batched actions, only keep facts that are actually safe to infer.
        # Do NOT try to localize which exact intermediate action caused the event.
        if batched_turn:
            if result.is_dead:
                self._mark_fire_death(new_pos, death_tmod20)
                self.episode_deaths += 1
                self.total_deaths_ever += 1
                self.current_pos = self.start_pos
                self.action_queue = []
                self.phase = 1
                self._advance_internal_clock(result.actions_executed)
                return

            if result.is_goal_reached:
                self.goal_pos = new_pos
                self._remember_position(new_pos)
                self._finish_episode(success=True)
                self.action_queue = []

                save_successful_replay()

                if self.phase == 1:
                    self.phase = 2
                    steps = len(self.optimal_path)
                    print(f"\nGoal found at {new_pos}!")
                    print(f"Replay path : {steps} steps ({steps//5 + 1} turns)")

                self._advance_internal_clock(result.actions_executed)
                return

            if result.is_confused:
                self.is_confused = not self.is_confused
                self.episode_confused += 1
                self.action_queue = []
                self.phase = 1

            # final position is trustworthy, intermediate causes are not
            self._remember_position(new_pos)

            # if something unexpected happened in a batched turn, drop back to cautious mode
            if result.wall_hits > 0 or result.teleported or result.is_confused or result.was_forced:
                print(
                    f"[REPLAY BREAK] "
                    f"pos={new_pos} "
                    f"wall_hits={result.wall_hits} "
                    f"teleported={result.teleported} "
                    f"forced={result.was_forced} "
                    f"confused={result.is_confused} "
                    f"actions_executed={result.actions_executed} "
                    f"remaining_replay={len(self.action_queue)}"
                )
                self.action_queue = []
                self.phase = 1

            self._advance_internal_clock(result.actions_executed)
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
            self._advance_internal_clock(result.actions_executed)
            return

        # ── DEATH ─────────────────────────────────────────────────────────────
        if result.is_dead:
            self._mark_fire_death(new_pos, death_tmod20)
            if self.phase == 1 and self.prev_pos and self.prev_action is not None:
                self._update_q(self.prev_pos, self.prev_action, -100, new_pos)
                self.q_table[new_pos[0], new_pos[1], :] = -200.0
            self.episode_deaths += 1
            self.total_deaths_ever += 1
            self.current_pos = self.start_pos
            self.action_queue = []
            self.phase = 1
            self._advance_internal_clock(result.actions_executed)
            return

        # ── TELEPORT ──────────────────────────────────────────────────────────
        if result.teleported and self.prev_pos and self.prev_action is not None:
            dr, dc = DELTAS[self.prev_action]
            pad = (self.prev_pos[0] + dr, self.prev_pos[1] + dc)
            if pad != new_pos:
                self.teleports[pad] = new_pos
                self.teleports[new_pos] = pad

            self._record_safe_move(self.prev_pos, self.prev_action, new_pos, teleported=True)

            if self.phase == 2:
                print(f"[REPLAY] teleport to {new_pos}")
            else:
                self.action_queue = []

        # ── CONFUSION ─────────────────────────────────────────────────────────
        if result.is_confused:
            self.is_confused = not self.is_confused
            self.confuse.add(new_pos)
            self.episode_confused += 1
            if self.phase == 1 and self.prev_pos and self.prev_action is not None:
                self._update_q(self.prev_pos, self.prev_action, -5, new_pos)

            if self.phase == 2:
                print(f"[REPLAY] confusion at {new_pos}")
            else:
                self.action_queue = []

        # ── DIRECTIONAL PUSH (Maze-gamma) ─────────────────────────────────────
        if result.was_forced and self.prev_pos and self.prev_action is not None:
            dr, dc = DELTAS[self.prev_action]
            push_cell = (self.prev_pos[0] + dr, self.prev_pos[1] + dc)
            self.push_tiles[push_cell] = result.forced_direction

            pdr, pdc = DELTAS[result.forced_direction]
            forced_dest = (push_cell[0] + pdr, push_cell[1] + pdc)
            if new_pos == push_cell:
                self.walls.add((push_cell[0], push_cell[1], result.forced_direction))
                if self._in_bounds(*forced_dest):
                    self.walls.add((forced_dest[0], forced_dest[1], INVERT[result.forced_direction]))
            else:
                self.safe_moves.add((push_cell[0], push_cell[1], result.forced_direction))

            if self.phase == 2:
                print(f"[REPLAY] push at {push_cell} dir={result.forced_direction}, continuing replay")
            else:
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

            save_successful_replay()

            if self.phase == 1:
                self.phase = 2
                steps = len(self.optimal_path)
                print(f"\nGoal found at {new_pos}!")
                print(f"Replay path : {steps} steps ({steps//5 + 1} turns)")

            self._record_safe_move(
                self.prev_pos,
                self.prev_action,
                new_pos,
                teleported=(result.teleported or result.was_forced)
            )
            self._advance_internal_clock(result.actions_executed)
            return

        # ── NORMAL MOVE ───────────────────────────────────────────────────────
        is_new = new_pos not in self.visited
        if self.phase == 1 and self.prev_pos and self.prev_action is not None:
            self._update_q(self.prev_pos, self.prev_action, +10 if is_new else -1, new_pos)
        self.visited.add(new_pos)
        self.episode_visited.add(new_pos)
        self.episode_cells.append(new_pos)
        self.current_pos = new_pos
        self._record_safe_move(
            self.prev_pos,
            self.prev_action,
            new_pos,
            teleported=(result.teleported or result.was_forced)
        )
        self._advance_internal_clock(result.actions_executed)

    # ─────────────────────────────────────────────────────────────────────────
    # PLAN TURN
    # ─────────────────────────────────────────────────────────────────────────

    def _plan_to_goal(self) -> List[int]:
        """
        Plan a route from current position to the known goal using the
        learned map. Prefers strict (safe-moves-only) A*, then relaxes.

        Returns [] if goal is unknown, current position is unknown, or no
        route can be found even under the relaxed model.
        """
        if self.goal_pos is None or self.current_pos is None:
            return []
        if self.current_pos == self.goal_pos:
            return []

        # Strict: only traverse edges we have confirmed safe by experience.
        path = self._astar(self.current_pos, self.goal_pos, require_safe=True)
        if path:
            return path

        # Relaxed: allow any edge that isn't a known wall or known-dead cell.
        # We may hit an untried wall or an unseen fire cell, but that's
        # recoverable (wall learning, respawn on death) -- far better than
        # wandering randomly forever with a known goal.
        path = self._astar(self.current_pos, self.goal_pos, require_safe=False)
        return path

    def plan_turn(self, last_result=None) -> List[int]:
        if last_result is not None:
            self._process_result(last_result)

        self.episode_turns += 1
        self.total_turns_ever += 1

        # Phase 2: replay known successful path, but batch only trusted segments
        if self.phase == 2:
            # If the replay queue ran out (or a prior turn cleared it because
            # something unexpected happened), try to replan directly to the
            # goal from wherever we are now. Staying in phase 2 avoids
            # collapsing into aimless BFS exploration.
            if not self.action_queue:
                replanned = self._plan_to_goal()
                if replanned:
                    self.action_queue = replanned

            if self.action_queue:
                trusted = self._trusted_prefix(self.action_queue, max_len=5)

                if trusted:
                    self.action_queue = self.action_queue[len(trusted):]
                    return self._submit(trusted)

                # If next replay segment is risky, replay one action at a time
                next_step = self.action_queue.pop(0)
                return self._submit([next_step])

            # Replay finished or failed AND replan found nothing usable.
            # Fall back to cautious exploration.
            self.phase = 1
            self.action_queue = []

        # Phase 1: epsilon-greedy RL exploration
        # With probability epsilon: exploit Q-table (pick best valid action).
        # Otherwise: fall back to Q-biased BFS to reach the nearest new cell.
        if not self.action_queue:
            if self.current_pos is not None and random.random() > self.epsilon:
                # ── Exploit: let the Q-table choose ──────────────────────────
                r, c = self.current_pos
                best_action = None
                best_q = -float("inf")
                for a in MOVE_ACTIONS:
                    if self._can_move(r, c, a):
                        q = float(self.q_table[r, c, a])
                        if q > best_q:
                            best_q = q
                            best_action = a
                if best_action is not None:
                    return self._submit([best_action])
            # ── Explore: Q-biased BFS toward nearest unvisited cell ───────────
            self.action_queue = self._bfs_explore()

        # Exploration exhausted but goal is known → head there directly.
        # Prevents the pathological "random walk until timeout" state that
        # triggers once the maze is fully mapped and all-cells-visited.
        if not self.action_queue and self.goal_pos is not None:
            self.action_queue = self._plan_to_goal()
            if self.action_queue:
                print(
                    f"[PHASE1 GOAL-SEEK] "
                    f"pos={self.current_pos} → goal={self.goal_pos} "
                    f"steps={len(self.action_queue)}"
                )

        if self.action_queue:
            return self._submit([self.action_queue.pop(0)])

        return self._submit([random.choice(MOVE_ACTIONS)])

    def _submit(self, desired_actions: List[int]) -> List[int]:
        desired_actions = desired_actions or [random.choice(MOVE_ACTIONS)]

        if self.current_pos is not None:
            self.prev_pos = self.current_pos
            self.prev_action = desired_actions[0] if desired_actions[0] in MOVE_ACTIONS else None

        self.last_planned_actions = list(desired_actions)
        self.episode_world_actions.extend(desired_actions)
        self.last_turn_start_tmod20 = self._current_tmod20()

        submitted = []
        for action in desired_actions:
            if action == ACTION_WAIT:
                submitted.append(ACTION_WAIT)
            elif self.is_confused:
                submitted.append(INVERT[action])
            else:
                submitted.append(action)

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
            "epsilon":          round(self.epsilon, 4),
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
            "deaths_mapped":    sum(len(cells) for cells in self.dead_cells_by_phase.values()),
            "teleports_mapped": len(self.teleports) // 2,
            "confuse_mapped":   len(self.confuse),
            "push_mapped":      len(self.push_tiles),
        }

    def print_metrics(self) -> None:
        m = self.get_metrics()
        print("\n" + "=" * 55)
        print("AGENT PERFORMANCE METRICS")
        print("=" * 55)
        print(f"  Phase              : {'SPEED RUN (A*)' if m['phase']==2 else 'EXPLORING (ε-greedy + BFS)'}")
        print(f"  Epsilon (ε)        : {m['epsilon']} (min={self.epsilon_min}, decay={self.epsilon_decay})")
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
    