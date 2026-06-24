import gymnasium as gym
import h5py
import json
import numpy as np
import os
from glob import glob
from mani_skill.utils.wrappers.record import RecordEpisode
import mani_skill.trajectory.utils as trajectory_utils
from mani_skill.envs.sapien_env import BaseEnv


def replay_trajectory(
    h5_path,
    json_path,
    env_id="TableTopCutting-v1",
    robot_uid="pkv",
    shader="default",
    headless=False
):
    print(f"\n=== Replaying {h5_path} ===")

 
    traj_data = h5py.File(h5_path, "r")
    available_keys = list(traj_data.keys())
    print(f"Available keys in h5: {available_keys}")

    with open(json_path, "r") as f:
        meta_data = json.load(f)

  
    render_mode = "rgb_array" if headless else "human"
    env = gym.make(
        env_id,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=render_mode,
        reward_mode="none",
        enable_shadow=True,
        viewer_camera_configs=dict(shader_pack=shader),
        robot_uids=robot_uid,
    )

    env = RecordEpisode(
        env,
        output_dir=os.path.dirname(h5_path) + "/replays",
        trajectory_name=os.path.basename(h5_path).replace(".h5", "_replay"),
        save_video=True,       
        save_trajectory=False,
        video_fps=30,
    )

    # --- episode ---
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

        # reset
        env.reset(**episode["reset_kwargs"])
        env_states_list = trajectory_utils.dict_to_list_of_dicts(data["env_states"])
        env.base_env.set_state_dict(env_states_list[0])


        for i, action in enumerate(np.array(data["actions"])):
            env.step(action)

       
            if not headless and i % 5 == 0:
                env.render()

    traj_data.close()
    env.close()
    print(f"Finished replay {h5_path}")


if __name__ == "__main__":
    demo_dir = "/home/por1329/maniskill/ManiSkill/demos/TableTopCutting-v1/teleop0827_4186"

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
                    env_id="TableTopCutting-v1",
                    robot_uid="pk",
                    shader="default",  
                    headless=False     
                )
            else:
                print(f"Warning: {json_path} not found, skipping.")
