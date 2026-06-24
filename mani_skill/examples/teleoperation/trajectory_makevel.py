import gymnasium as gym
import numpy as np
import h5py
import json
from dataclasses import dataclass
from typing import Annotated
import tyro
import sapien.core as sapien
import sapien.utils.viewer

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.record import RecordEpisode
import mani_skill.trajectory.utils as trajectory_utils
import random
from copy import deepcopy
import os


# --- constants ---
OPEN = 1
CLOSED = -1

# --- planners ---
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_knife import (
    PandaKnifeMotionPlanningSolver,
)


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "TableTopCuttingct"
    obs_mode: str = "none"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "pkour"
    """Supported: panda, panda_wristcam, panda_stick, pk (knife)"""
    record_dir: str = "demos"
    save_video: bool = False
    viewer_shader: str = "rt-fast"
    video_saving_shader: str = "rt-fast"


def parse_args() -> Args:
    return tyro.cli(Args)


def main(args: Args):
    output_dir = f"{args.record_dir}/{args.env_id}/teleop0827_{random.randint(1000, 9999)}/"
    print(output_dir)

    # ----------------------------------------------------------- #
    # 🌟 변경점 1: control_mode를 "pd_joint_pos_vel"로 변경
    # ----------------------------------------------------------- #
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos_vel",  # ⬅️ 이 부분을 수정했습니다.
        render_mode="human",
        reward_mode="none",
        enable_shadow=False,
        human_render_camera_configs=dict(shader_pack="default"),
        robot_uids=args.robot_uid,
    )
    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="teleoperation via the click+drag system",
    )

    num_trajs = 0
    seed = 0

    env.reset(seed=seed, options=dict(save_trajectory=True))

    while True:
        print(f"Collecting trajectory {num_trajs + 1}, seed={seed}")
        code = solve(env, debug=False, vis=True)

        if code == "quit":
            num_trajs += 1
            break
        elif code == "continue":
            seed += 1
            num_trajs += 1
            state = env.get_state_dict()
            env.reset(seed=seed)
            env.set_state_dict(state)
            continue
        elif code == "restart":
            env.reset(seed=seed, options=dict(save_trajectory=True))
            
    h5_file_path = env._h5_file.filename
    json_file_path = env._json_path
    env.close()
    del env
    print(f"Trajectories saved to {h5_file_path}")

    if args.save_video:
        print(f"Saving videos to {output_dir}")
        trajectory_data = h5py.File(h5_file_path, "r")
        with open(json_file_path, "r") as f:
            json_data = json.load(f)

        env = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            control_mode="pd_joint_pos_vel", # ⬅️ 비디오 재생 시에도 컨트롤러 모드를 일치시킵니다.
            render_mode="human",
            reward_mode="none",
            enable_shadow=True,
            viewer_camera_configs=dict(shader_pack=args.viewer_shader),
            robot_uids=args.robot_uid,
        )
        env = RecordEpisode(
            env,
            output_dir=output_dir,
            trajectory_name="trajectory",
            save_video=True,
            info_on_video=False,
            save_trajectory=False,
            video_fps=30,
        )

        for episode in json_data["episodes"]:
            traj_id = f"traj_{episode['episode_id']}"
            data = trajectory_data[traj_id]
            env.reset(**episode["reset_kwargs"])
            env_states_list = trajectory_utils.dict_to_list_of_dicts(data["env_states"])
            env.base_env.set_state_dict(env_states_list[0])

            for action in np.array(data["actions"]):
                env.step(action)

        trajectory_data.close()
        env.close()
        del env


def solve(env: BaseEnv, debug=False, vis=False):
    # ----------------------------------------------------------- #
    # 🌟 변경점 2: pd_joint_pos_vel 컨트롤 모드도 허용
    # ----------------------------------------------------------- #
    assert env.unwrapped.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"]

    if env.unwrapped.robot_uids in ["panda_stick", "pk","pkv"]:
        planner = PandaKnifeMotionPlanningSolver(
            env, debug=debug, vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            print_env_info=False,
            joint_acc_limits=0.5,
            joint_vel_limits=0.5,
            lock_gripper=True,
            locked_gripper_value=CLOSED,
        )
        robot_has_gripper = True
    elif env.unwrapped.robot_uids in ["panda", "panda_wristcam"]:
        planner = PandaArmMotionPlanningSolver(
            env, debug=debug, vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_acc_limits=0.5,
            joint_vel_limits=0.5,
        )
        robot_has_gripper = True
    else:
        raise ValueError(f"Unsupported robot_uids: {env.unwrapped.robot_uids}")

    viewer = env.render_human()
    transform_window = None
    for plugin in viewer.plugins:
        try:
            import sapien.utils.viewer.viewer as vmod
            if isinstance(plugin, vmod.TransformWindow):
                transform_window = plugin
                break
        except Exception:
            pass

    def print_controls():
        print("""[Controls]
            h: help
            n: ghost pose record
            g: gripper control
            c: save action and next
            q: quit"""
        )

    def select_panda_tcp():
        tcp_link = sapien_utils.get_obj_by_name(env.agent.robot.links, "tcp")
        if tcp_link is not None and hasattr(tcp_link, "_objs") and tcp_link._objs:
            viewer.select_entity(tcp_link._objs[0].entity)

    select_panda_tcp()
    print_controls()
    gripper_open = True

    while not viewer.closed:
        if transform_window is not None:
            transform_window.enabled = True

        env.render_human()
        execute_current_pose = False

        if viewer.window.key_press("h"):
            print_controls()
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
        elif viewer.window.key_press("n"):
            execute_current_pose = True
        elif viewer.window.key_press("p"):
            obs = env.get_obs()
            if "sensor_data" not in obs:
                print("No sensor_data in obs")
            else:
                for cam_name, file_name in [
                    ("hand_camera", "knife_cam.png"),
                    ("base_camera", "front_cam.png"),
                    ("top_camera", "top_cam.png"),
                    ("right_cam", "right_cam.png"),
                    ("left_cam", "left_cam.png"),
                ]:
                    if cam_name in obs["sensor_data"]:
                        img = obs["sensor_data"][cam_name]["rgb"]
                        img = img.cpu().numpy()

                        # 배치 차원 제거
                        if img.ndim == 4 and img.shape[0] == 1:
                            img = img[0]

                        # 값이 0~1 float이면 0~255 uint8로 변환
                        if img.dtype != np.uint8:
                            img = (img * 255).clip(0, 255).astype(np.uint8)

                        print(f"{cam_name} final shape:", img.shape)

                        save_path = os.path.join("/home/por1329/maniskill/ManiSkill/camera/", file_name)
                        import imageio
                        imageio.imwrite(save_path, img)
                        print(f"[Saved] {save_path}")

        elif viewer.window.key_press("g") and robot_has_gripper:
            if gripper_open:
                gripper_open = False
                _, reward, _, _, info = planner.close_gripper()
            else:
                gripper_open = True
                _, reward, _, _, info = planner.open_gripper()
            print(f"Reward: {reward}, Info: {info}")

        if execute_current_pose:
            gizmo_pose = getattr(transform_window, "_gizmo_pose", env.unwrapped.agent.tcp.pose)
            result = None
            if env.unwrapped.robot_uids in ["pk", "panda_stick"]:
                result = planner.move_to_pose_with_screw(
                    gizmo_pose, dry_run=True
                )
                if result == -1 or result["status"] != "Success":
                    print("screw plan failed → fallback to RRTConnect")
                    result = planner.move_to_pose_with_RRTConnect(
                        gizmo_pose, dry_run=True
                    )
            else:
                result = planner.move_to_pose_with_RRTConnect(
                    gizmo_pose, dry_run=True
                )

            if result != -1 and len(result["position"]) < 500:
                _, reward, _, _, info = planner.follow_path(result)
                print(f"Reward: {reward}, Info: {info}")
            else:
                print("Plan failed or too long.")
            execute_current_pose = False

    return "continue"


if __name__ == "__main__":
    main(parse_args())