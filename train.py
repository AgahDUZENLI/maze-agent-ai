import numpy as np
import time
from agent import HybridAgent
from environment import MazeEnvironment
from visualizer import MazeVisualizer

# ── Config ───────────────────────────────────────────────────────────────────
MAZE_PATH   = "MAZE_0.png"
HAZARD_PATH = "MAZE_1.png"
VIZ_DIR     = "viz"
NUM_EPISODES = 5
MAX_TURNS    = 10_000
SAVE_PATH    = "q_table_alpha.npy"


def train():
    print("=" * 55)
    print("SILENT CARTOGRAPHER — Q-Learning Training")
    print("=" * 55)

    env   = MazeEnvironment(MAZE_PATH, HAZARD_PATH)
    agent = HybridAgent()

    print("fresh agent check")
    print("walls:", len(agent.walls))
    print("teleports:", len(agent.teleports))
    print("confuse:", len(agent.confuse))
    print("visited:", len(agent.visited))
    print("goal_pos:", agent.goal_pos)
    print("phase:", agent.phase)
    print("q_table_nonzero:", np.count_nonzero(agent.q_table))
    
    viz   = MazeVisualizer(MAZE_PATH)

    start_time = time.time()

    for episode in range(1, NUM_EPISODES + 1):

        # ── Reset ─────────────────────────────────────────────────────────────
        start_pos   = env.reset()
        agent.reset_episode(start_pos)
        last_result = None
        success     = False

        # Track path and deaths this episode for visualization
        path_taken  = [start_pos]   # ordered list of positions
        death_cells = []            # where agent died

        # ── Episode loop ──────────────────────────────────────────────────────
        for turn in range(MAX_TURNS):

            actions     = agent.plan_turn(last_result)
            last_result = env.step(actions)

            # Track where agent is now
            path_taken.append(last_result.current_position)

            # Track deaths
            if last_result.is_dead:
                death_cells.append(last_result.current_position)
                agent.current_pos = env.start

            # Goal reached
            if last_result.is_goal_reached:
                # CRITICAL: process this result NOW so goal_pos gets saved.
                # Normally plan_turn() processes last_result at the start of
                # the next turn — but there IS no next turn when goal is reached.
                agent._process_result(last_result)
                success = True
                agent.all_path_lengths[-1] = env.get_episode_stats()["path_length"]
                print(f"  [DEBUG] Goal reached! turn={turn} pos={last_result.current_position} agent.goal_pos={agent.goal_pos}")
                break

        # ── Finish episode ────────────────────────────────────────────────────
        if not success and (last_result is None or not last_result.is_goal_reached):
            agent.finish_episode_timeout()

        # ── Print stats ───────────────────────────────────────────────────────
        stats   = env.get_episode_stats()
        m       = agent.get_metrics()
        elapsed = time.time() - start_time
        status  = "✓ SUCCESS" if success else "✗ timeout"

        print(
            f"ep {episode:>4} | {status} | "
            f"turns={stats['turns_taken']:>6} | "
            f"deaths={stats['deaths']:>2} | "
            f"cells={m['unique_cells']:>4} | "
            f"goal={'YES' if m['goal_found'] else 'no':<3} | "
            f"success={m['success_rate']:>5.1f}% | "
            f"t={elapsed:.0f}s"
        )

        # ── Save visualization ────────────────────────────────────────────────
        viz.save_episode(
            episode_num = episode,
            agent       = agent,
            env         = env,
            path_taken  = path_taken,
            death_cells = death_cells,
            output_dir  = VIZ_DIR,
        )

    # ── Final report ──────────────────────────────────────────────────────────
    agent.print_metrics()

    # ── Save Q-table ──────────────────────────────────────────────────────────
    np.save(SAVE_PATH, agent.q_table)
    print(f"\nQ-table saved → {SAVE_PATH}")
    print(f"Episode images → {VIZ_DIR}/")

    return agent


if __name__ == "__main__":
    train()
