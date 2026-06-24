import gymnasium as gym
import h5py
import json
import numpy as np
import os
from glob import glob
from mani_skill.utils.wrappers.record import RecordEpisode
import mani_skill.trajectory.utils as trajectory_utils
from mani_skill.envs.sapien_env import BaseEnv
import mani_skill.envs
import mani_skill
print("mani_skill imported from:", mani_skill.__file__)
from mani_skill.sensors.camera import CameraConfig
import gymnasium as gym
print([k for k in gym.envs.registry.keys() if "bananacut" in k])
# print([k for k in gym.envs.registry.keys() if "Cutting" in k])
# print(gym.envs.registry.keys())

def replay_trajectory(
    h5_path,
    json_path,
    env_id="bananacut",
    robot_uid="pkour",
    shader="default",
    save_video=True,
):
    print(">>> DEBUG env_id =", env_id)
    print(f"\n=== Replaying {h5_path} ===")

    traj_data = h5py.File(h5_path, "r")
    available_keys = list(traj_data.keys())
    print(f"Available keys in h5: {available_keys}")

    with open(json_path, "r") as f:
        meta_data = json.load(f)
    env = gym.make(
        env_id,
        obs_mode="rgbd",
        control_mode="pd_joint_pos_vel",
        render_mode="rgb_array",   # ✅ headless 모드 (GUI 안 띄움)
        reward_mode="none",
        enable_shadow=True,
        viewer_camera_configs=dict(shader_pack=shader),
        robot_uids=robot_uid,
    )
    env = RecordEpisode(
        env,
        output_dir=os.path.dirname(h5_path) + "/replays",
        trajectory_name=os.path.basename(h5_path).replace(".h5", "_replay"),
        save_video=save_video,    # ✅ mp4 저장
        save_trajectory=False,
        video_fps=30,
        save_camera="base_camera"
    )

    # --- episode별 재생 ---
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

        # actions play
        for step_idx, action in enumerate(np.array(data["actions"])):
            obs = env.get_obs()
            if "sensor_data" in obs:
                for cam_name in ["hand_camera", "base_camera", "top_camera", "right_cam", "left_cam"]:
                # for cam_name in ["right_cam", "left_cam"]:
                    if cam_name in obs["sensor_data"]:
                        img = obs["sensor_data"][cam_name]["rgb"]
                        img = img.cpu().numpy()
                        if img.ndim == 4 and img.shape[0] == 1:
                            img = img[0]
                        if img.dtype != np.uint8:
                            img = (img * 255).clip(0, 255).astype(np.uint8)

                        # ✅ 프레임 인덱스 포함 저장
                        save_path = os.path.join(
                            "/home/por1329/maniskill/ManiSkill/camera/",
                            f"{cam_name}_{step_idx:05d}.png"
                        )
                        import imageio
                        imageio.imwrite(save_path, img)
            env.step(action)

    traj_data.close()
    env.close()
    print(f"Finished replay {h5_path}")


if __name__ == "__main__":
    demo_dir = "/home/por1329/maniskill/ManiSkill/demos/bananacut/teleop21_3351"

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
                    env_id="bananacut",
                    robot_uid="pkour",
                    shader="default",
                    save_video=True,
                )
            else:
                print(f"Warning: {json_path} not found, skipping.")
