import os
import shutil
import numpy as np
import time

from agent import HybridAgent
from environment import MazeEnvironment
from visualizer import MazeVisualizer


# ── SELECT MAZE HERE ───────────────────────────────
MAZE_NAME = "alpha"   # change: "alpha" / "beta" / "gamma"

EPISODES       = 5
TEST_EPISODES  = 5
MAX_TURNS      = 10_000
GIF_SKIP       = 200
GIF_FPS        = 12


# ── PATHS ──────────────────────────────────────────
MAZE_DIR  = os.path.join("TestMazes", f"maze-{MAZE_NAME}")
MAZE_PATH = os.path.join(MAZE_DIR, "MAZE_0.png")

RUN_DIR   = os.path.join("runs", MAZE_NAME)
VIZ_DIR   = os.path.join(RUN_DIR, "viz")
SAVE_PATH = os.path.join(RUN_DIR, "q_table.npy")
REPLAY_DIR = os.path.join(MAZE_DIR, "results", "replays")


# ───────────────────────────────────────────────────
# RUN EPISODES
# ───────────────────────────────────────────────────
def run_episodes(env, agent, viz, num_episodes, mode, start_time):

    results = []

    for ep in range(1, num_episodes + 1):

        start_pos = env.reset()
        agent.reset_episode(start_pos)

        last_result = None
        success = False

        path_taken = [start_pos]
        death_cells = []
        current_segment = [start_pos]

        for turn in range(MAX_TURNS):

            actions = agent.plan_turn(last_result)
            last_result = env.step(actions)

            path_taken.append(last_result.current_position)
            current_segment.append(last_result.current_position)

            if last_result.is_dead:
                death_cells.append(last_result.current_position)
                agent.current_pos = env.start
                current_segment = [env.start]

            # ── CAPTURE FRAME ───────────────────────
            viz.capture_frame(
                agent=agent,
                env=env,
                path_so_far=current_segment,
                deaths_so_far=death_cells,
                episode_num=ep,
            )

            if last_result.is_goal_reached:
                agent._process_result(last_result)
                success = True
                break

        if not success:
            agent.finish_episode_timeout()

        stats = env.get_episode_stats()
        m = agent.get_metrics()

        print(
            f"[{mode.upper()}] ep {ep:>3} | "
            f"{'SUCCESS' if success else 'timeout'} | "
            f"turns={stats['turns_taken']:>5} | "
            f"deaths={stats['deaths']:>2} | "
            f"cells={m['unique_cells']:>4} | "
            f"goal={'YES' if m['goal_found'] else 'no':<3} | "
            f"e={m['epsilon']:.3f}"
        )

        # ── SAVE GIF + PNG ────────────────────────
        viz.save_episode(
            episode_num=ep,
            agent=agent,
            env=env,
            path_taken=path_taken,
            death_cells=death_cells,
            output_dir=os.path.join(VIZ_DIR, mode),
        )

        replay_path = env.export_replay(
            os.path.join(REPLAY_DIR, f"{mode}_episode_{ep:03d}.json"),
            agent_overlay={"metrics": agent.get_metrics()},
            dynamic_fire=True,
        )
        shutil.copyfile(replay_path, os.path.join(REPLAY_DIR, f"latest_{mode}.json"))
        shutil.copyfile(replay_path, os.path.join(REPLAY_DIR, "latest.json"))
        print(f"  replay: {replay_path}")

        results.append({
            "episode": ep,
            "success": success,
            "turns": stats["turns_taken"],
            "deaths": stats["deaths"],
        })

    return results


# ───────────────────────────────────────────────────
# REPORT
# ───────────────────────────────────────────────────
def print_report(results, label):
    print("\n" + "=" * 50)
    print(label)
    print("=" * 50)

    successes = [r for r in results if r["success"]]

    print(f"Episodes:     {len(results)}")
    print(f"Success rate: {len(successes) / len(results) * 100:.1f}%")
    print(f"Avg turns:    {np.mean([r['turns'] for r in results]):.1f}")
    print(f"Avg deaths:   {np.mean([r['deaths'] for r in results]):.1f}")
    print("=" * 50)


# ───────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────
def main():
    print(f"\n=== RUNNING MAZE: {MAZE_NAME.upper()} ===\n")

    os.makedirs(os.path.join(VIZ_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(VIZ_DIR, "test"), exist_ok=True)

    env = MazeEnvironment(MAZE_NAME)
    agent = HybridAgent()

    viz = MazeVisualizer(
        MAZE_PATH,
        gif_fps=GIF_FPS,
        gif_skip=GIF_SKIP
    )

    # ── Load Q-table ─────────────────────────────
    if os.path.exists(SAVE_PATH):
        agent.q_table = np.load(SAVE_PATH)
        print("Loaded existing Q-table")
    else:
        print("Starting fresh Q-table")

    start_time = time.time()

    # ── TRAIN ────────────────────────────────────
    print("\n--- TRAIN ---\n")

    train_results = run_episodes(
        env, agent, viz,
        EPISODES,
        mode="train",
        start_time=start_time
    )

    np.save(SAVE_PATH, agent.q_table)
    print_report(train_results, "TRAIN RESULTS")

    # ── TEST ─────────────────────────────────────
    print("\n--- TEST ---\n")

    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0  # deterministic test

    test_results = run_episodes(
        env, agent, viz,
        TEST_EPISODES,
        mode="test",
        start_time=start_time
    )

    agent.epsilon = saved_epsilon

    print_report(test_results, "TEST RESULTS")

    print("\nDone. Check runs/ folder.\n")


# ───────────────────────────────────────────────────
if __name__ == "__main__":
    main()
