import gymnasium as gym
import h5py
import json
import numpy as np
import os
from glob import glob
from mani_skill.utils.wrappers.record import RecordEpisode
import mani_skill.trajectory.utils as trajectory_utils
import mani_skill.envs  # 환경 등록 보장


def replay_trajectory(
    h5_path,
    json_path,
    env_id="bananacut",
    robot_uid="pkours",
    shader="default",
    headless=False
):
    print(f"\n=== Replaying {h5_path} ===")

    # h5 trajectory data
    traj_data = h5py.File(h5_path, "r")
    available_keys = list(traj_data.keys())
    print(f"Available keys in h5: {available_keys}")

    # meta data (episode info)
    with open(json_path, "r") as f:
        meta_data = json.load(f)

    # make environment
    render_mode = "rgb_array" if headless else "human"
    env = gym.make(
        env_id,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=render_mode,
        reward_mode="none",
        enable_shadow=True,
        robot_uids=robot_uid,
    )

    # wrap with recorder
    env = RecordEpisode(
        env,
        output_dir=os.path.dirname(h5_path) + "/replays",
        trajectory_name=os.path.basename(h5_path).replace(".h5", "_replay"),
        save_video=True,
        save_trajectory=False,
        video_fps=30,
    )

    # --- replay each episode ---
    for epi_idx, episode in enumerate(meta_data["episodes"]):
        traj_id = f"traj_{episode['episode_id']}"
        if traj_id not in available_keys:
            if epi_idx < len(available_keys):
                traj_id = available_keys[epi_idx]
                print(f"[WARN] {traj_id} not in h5, fallback to {traj_id}")
            else:
                print(f"[ERROR] Cannot find matching traj for episode {epi_idx}, skipping")
                continue

        data = traj_data[traj_id]

        # reset environment
        env.reset(**episode["reset_kwargs"])
        env_states_list = trajectory_utils.dict_to_list_of_dicts(data["env_states"])

        # unwrap to access set_state_dict
        base_env = env.unwrapped
        base_env.set_state_dict(env_states_list[0])

        # step through trajectory
        for i, action in enumerate(np.array(data["actions"])):
            env.step(action)
            if not headless and i % 5 == 0:
                env.render()

    traj_data.close()
    env.close()
    print(f"Finished replay {h5_path}")


if __name__ == "__main__":
    demo_dir = "/home/por1329/maniskill/ManiSkill/demos/bananacut/merge/merged_trajectories.h5"

    h5_files = sorted(glob(os.path.join(demo_dir, "*.h5")))
    if not h5_files:
        print(f"No h5 files found in {demo_dir}")
    else:
        for h5_path in h5_files:
            json_path = h5_path.replace(".h5", ".json")
            if os.path.exists(json_path):
                replay_trajectory(
                    h5_path,
                    json_path,
                    env_id="TableTopCuttingct",
                    robot_uid="pkv",   # 일관성 유지
                    shader="default",
                    headless=False
                )
            else:
                print(f"Warning: {json_path} not found, skipping.")
