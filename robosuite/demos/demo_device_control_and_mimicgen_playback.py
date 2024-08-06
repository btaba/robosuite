"""
Record trajectory data with the DataCollectionWrapper wrapper and play them back.

Example:
    $ python demo_collect_and_playback_data.py --environment Lift
"""

import argparse
import os
from glob import glob

import numpy as np
import math
import robosuite as suite
from robosuite.wrappers import DataCollectionWrapper
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper
import robosuite.utils.transform_utils as T


def collect_device_trajectory(env, args, timesteps=1000):
    """Collect device trajectory.
    Args:
        env (MujocoEnv): environment instance to collect trajectories from
        timesteps(int): how many environment timesteps to run for a given trajectory
    """
    obs = env.reset()

    # Setup rendering
    cam_id = 0
    num_cam = len(env.sim.model.camera_names)
    env.render()

    # Initialize variables that should the maintained between resets
    last_grasp = 0

    # Initialize device control
    device.start_control()
    while True:
        # Set active robot
        active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]

        # Get the newest action
        action, grasp = input2action(
            device=device, robot=active_robot, active_arm=args.arm, env_configuration=args.config
        )

        # If action is none, then this a reset so we should break
        if action is None:
            break

        # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
        # toggle arm control and / or camera viewing angle if requested
        if last_grasp < 0 < grasp:
            if args.switch_on_grasp:
                args.arm = "left" if args.arm == "right" else "right"
            if args.toggle_camera_on_grasp:
                cam_id = (cam_id + 1) % num_cam
                env.viewer.set_camera(camera_id=cam_id)
        # Update last grasp
        last_grasp = grasp

        # Fill out the rest of the action space if necessary
        rem_action_dim = env.action_dim - action.size
        if rem_action_dim > 0:
            # Initialize remaining action space
            rem_action = np.zeros(rem_action_dim)
            # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
            if args.arm == "right":
                action = np.concatenate([action, rem_action])
            elif args.arm == "left":
                action = np.concatenate([rem_action, action])
            else:
                # Only right and left arms supported
                print(
                    "Error: Unsupported arm specified -- "
                    "must be either 'right' or 'left'! Got: {}".format(args.arm)
                )
        elif rem_action_dim < 0:
            # We're in an environment with no gripper action space, so trim the action space to be the action dim
            action = action[: env.action_dim]

        # Step through the simulation and render
        obs, reward, done, info = env.step(action)
        env.render()



def playback_trajectory(env, ep_dir):
    """Playback data from an episode.

    Args:
        env (MujocoEnv): environment instance to playback trajectory in
        ep_dir (str): The path to the directory containing data for an episode.
    """
    env.reset()

    state_paths = os.path.join(ep_dir, "state_*.npz")

    # read states back, load them one by one, and render
    t = 0
    for state_file in sorted(glob(state_paths)):
        print(state_file)
        dic = np.load(state_file)
        states = dic["states"]
        for state in states:
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env.render()
            t += 1
            if t % 100 == 0:
                print(t)


def quat2axisangle(quat):
    """
    Converts (x, y, z, w) quaternion to axis-angle format.
    Returns a unit vector direction and an angle.

    NOTE: this differs from robosuite's function because it returns
          both axis and angle, not axis * angle.
    """

    # conversion from axis-angle to quaternion:
    #   qw = cos(theta / 2); qx, qy, qz = u * sin(theta / 2)

    # normalize qx, qy, qz by sqrt(qx^2 + qy^2 + qz^2) = sqrt(1 - qw^2)
    # to extract the unit vector

    # clipping for scalar with if-else is orders of magnitude faster than numpy
    if quat[3] > 1.:
        quat[3] = 1.
    elif quat[3] < -1.:
        quat[3] = -1.

    den = np.sqrt(1. - quat[3] * quat[3])
    if math.isclose(den, 0.):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3), 0.

    return quat[:3] / den, 2. * math.acos(quat[3])


def axisangle2quat(axis, angle):
    """
    Converts axis-angle to (x, y, z, w) quat.

    NOTE: this differs from robosuite's function because it accepts
          both axis and angle as arguments, not axis * angle.
    """

    # handle zero-rotation case
    if math.isclose(angle, 0.):
        return np.array([0., 0., 0., 1.])

    # make sure that axis is a unit vector
    assert math.isclose(np.linalg.norm(axis), 1., abs_tol=1e-3)

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.)
    q[:3] = axis * np.sin(angle / 2.)
    return q


def quat_slerp(q1, q2, tau):
    """
    Adapted from robosuite.
    """
    if tau == 0.0:
        return q1
    elif tau == 1.0:
        return q2
    d = np.dot(q1, q2)
    if abs(abs(d) - 1.0) < np.finfo(float).eps * 4.:
        return q1
    if d < 0.0:
        # invert rotation
        d = -d
        q2 *= -1.0
    angle = math.acos(np.clip(d, -1, 1))
    if abs(angle) < np.finfo(float).eps * 4.:
        return q1
    isin = 1.0 / math.sin(angle)
    q1 = q1 * math.sin((1.0 - tau) * angle) * isin
    q2 = q2 * math.sin(tau * angle) * isin
    q1 = q1 + q2
    return q1


def interpolate_rotations(R1, R2, num_steps, axis_angle=True):
    """
    Interpolate between 2 rotation matrices. If @axis_angle, interpolate the axis-angle representation
    of the delta rotation, else, use slerp.

    NOTE: I have verified empirically that both methods are essentially equivalent, so pick your favorite.
    """
    if axis_angle:
        # delta rotation expressed as axis-angle
        delta_rot_mat = R2.dot(R1.T)
        delta_quat = T.mat2quat(delta_rot_mat)
        delta_axis, delta_angle = quat2axisangle(delta_quat)

        # fix the axis, and chunk the angle up into steps
        rot_step_size = delta_angle / num_steps

        # convert into delta rotation matrices, and then convert to absolute rotations
        if delta_angle < 0.05:
            # small angle - don't bother with interpolation
            rot_steps = np.array([R2 for _ in range(num_steps)])
        else:
            delta_rot_steps = [T.quat2mat(axisangle2quat(delta_axis, i * rot_step_size)) for i in range(num_steps)]
            rot_steps = np.array([delta_rot_steps[i].dot(R1) for i in range(num_steps)])
    else:
        q1 = T.mat2quat(R1)
        q2 = T.mat2quat(R2)
        rot_steps = np.array([T.quat2mat(quat_slerp(q1, q2, tau=(float(i) / num_steps))) for i in range(num_steps)])
    
    # add in endpoint
    rot_steps = np.concatenate([rot_steps, R2[None]], axis=0)

    return rot_steps


def interpolate_positions(p1, p2, num_steps):
    # linear interpolation of positions
    pos_step_size = (p2 - p1) / num_steps
    grid = np.arange(num_steps).astype(np.float64)
    return np.array([p1 + grid[i] * pos_step_size for i in range(num_steps)])


def interpolate_poses(p1, p2, num_steps):
    pos1, rot1 = p1[:3, 3], p1[:3, :3]
    pos2, rot2 = p2[:3, 3], p2[:3, :3]
    pos = interpolate_positions(pos1, pos2, num_steps)
    rot = interpolate_rotations(rot1, rot2, num_steps)

    poses = []
    for p, r in zip(pos, rot):
        pose = np.eye(4)
        pose[:3, 3] = p
        pose[:3, :3] = r
        poses.append(pose)
    return poses


def to_robot_action(curr_obs, pose, gripper):
    # current position and rotation
    curr_pos = curr_obs["robot0_eef_pos"]
    curr_quat = curr_obs["robot0_eef_quat"]
    curr_rot = T.quat2mat(curr_quat)

    target_pos = pose[:3, 3]
    target_rot = pose[:3, :3]

    max_dpos, max_drot = 1., 1.

    # normalized delta position action
    delta_position = target_pos - curr_pos
    delta_position = np.clip(delta_position / max_dpos, -1., 1.)

    # normalized delta rotation action
    delta_rot_mat = target_rot.dot(curr_rot.T)
    delta_quat = T.mat2quat(delta_rot_mat)
    delta_rotation = T.quat2axisangle(delta_quat)
    delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)

    return np.concatenate([delta_position, delta_rotation, gripper])



def generate_trajectory(env, ep_dir):
    """Generate trajectory using mimicgen.

    Args:
        env (MujocoEnv): environment instance to playback trajectory in
        ep_dir (str): The path to the directory containing data for an episode.
    """
    # env reset will randomize stuff for us...
    env.reset()

    state_paths = os.path.join(ep_dir, "state_*.npz")
    qpos, src_action = [], []
    env_obs = []
    for state_file in sorted(glob(state_paths)):
        dic = np.load(state_file, allow_pickle=True)
        # skip time
        qpos.extend(dic["states"][1:env.sim.model.nq + 1])
        src_action.extend([a["actions"] for a in dic["action_infos"]])
        for state in dic["states"]:
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env_obs.append(env.unwrapped._get_observations(force_update=True))

    curr_obs = env.reset()

    # subtask 1, pickup cube
    # subtask 2, place cube
    # subtask 3, rest of trajectory
    subtasks = [
        ("cube", 1),
        ("cube", -1),
        (None, -1)
    ]
    # split source trajectory
    src_subtask_bounds, t = [], 0
    for obj, gripper in subtasks[:-1]:
        act, obs = src_action[t], env_obs[t]
        while gripper != int(act[-1]) and t < len(src_action):
            act, obs = src_action[t], env_obs[t]
            t += 1
        src_subtask_bounds.append(max(t - 1, 0))

    # import IPython; IPython.embed()

    t_src = 0
    last_waypoint = T.make_pose(
        curr_obs["robot0_eef_pos"],
        T.quat2mat(curr_obs["robot0_eef_quat"])
    )
    for idx, (obj, gripper) in enumerate(subtasks):
        print('Subtask: ', subtasks[idx])

        until_t_src = src_subtask_bounds[idx] if obj else len(src_action)
        gripper_act = [a[-1] for a in src_action[t_src:until_t_src]]

        target_eef_poses = []
        if obj:
            # assume object is stationary
            obj_pos, obj_quat = env_obs[t_src][f"{obj}_pos"], env_obs[t_src][f"{obj}_quat"]
            obj_pose = T.make_pose(obj_pos, T.quat2mat(obj_quat))
            curr_obj_pos, curr_obj_quat = curr_obs[f"{obj}_pos"], curr_obs[f"{obj}_quat"]
            curr_obj_pose = T.make_pose(curr_obj_pos, T.quat2mat(curr_obj_quat))

            target_eef_poses = []
            for t in range(t_src, until_t_src):
                eef_pose = T.make_pose(
                    env_obs[t]["robot0_eef_pos"],
                    T.quat2mat(env_obs[t]["robot0_eef_quat"])
                )
                eef_pose_in_obj_frame = T.pose_in_A_to_pose_in_B(
                    eef_pose, T.pose_inv(obj_pose))
                target_eef_pose = T.pose_in_A_to_pose_in_B(
                    eef_pose_in_obj_frame, curr_obj_pose)

                target_eef_poses.append(target_eef_pose)
        else:
            for t in range(t_src, until_t_src):
                target_eef_poses.append(T.make_pose(
                    env_obs[t]["robot0_eef_pos"],
                    T.quat2mat(env_obs[t]["robot0_eef_quat"])
                ))

        full_traj = []

        # interpolate from last waypoint to the first target_eef_pose
        interp_steps = 100
        if last_waypoint is not None and interp_steps:
            interp_poses = interpolate_poses(last_waypoint, target_eef_poses[0], interp_steps)
            gripper_act = [gripper_act[0]] * interp_steps + gripper_act
            full_traj.extend(interp_poses)

        # merge in the target_eef_pose into interpolated pose
        full_traj.extend(target_eef_poses)

        # pad with last
        pad_steps = 100
        last_pose = target_eef_poses[-1]
        full_traj.extend([last_pose] * pad_steps)
        gripper_act.extend([gripper_act[-1]] * pad_steps)

        assert len(gripper_act) == len(full_traj)
        # execute the trajectory
        for ti, (pose, grip) in enumerate(zip(full_traj, gripper_act)):
            if ti == interp_steps - 1:
                print('Done interp')
            if ti == interp_steps + len(target_eef_poses) - 1:
                print('Done traj')
            act = to_robot_action(curr_obs, pose, grip[None])
            curr_obs, *_ = env.step(act)
            env.render()
        print('Done pad')

        # store the last waypoint
        last_waypoint = full_traj[-1]
        t_src = until_t_src

    # # simply replaying action should fail
    # t = 0
    # for act in action:
    #     env.step(act)
    #     env.render()
    #     t += 1
    #     if t % 100 == 0:
    #         print(t)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument("--directory", type=str, default="/tmp/")
    parser.add_argument("--load-from", type=str, default="/tmp/ep_1722905411_536671")
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    # Import controller config for EE IK or OSC (pos/ori)
    if args.controller == "ik":
        controller_name = "IK_POSE"
    elif args.controller == "ik_mj":
        controller_name = "IK_MJ_POSE"
    elif args.controller == "osc":
        controller_name = "OSC_POSE"
    else:
        print("Error: Unsupported controller specified. Must be either 'ik' or 'osc'!")
        raise ValueError

    # Get controller config
    controller_config = load_controller_config(default_controller=controller_name)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    # Create environment
    orig_env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )

    # wrap the environment with data collection wrapper
    data_directory = args.directory
    env = DataCollectionWrapper(orig_env, data_directory)
    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)
    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # testing to make sure multiple env.reset calls don't create multiple directories
    env.reset()
    env.reset()
    env.reset()

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback(device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    if not args.load_from:
        # collect some data
        print("Collecting some device data...")
        collect_device_trajectory(env, args, timesteps=args.timesteps)
    else:
        print("Loading device data from ", args.load_from)

    # playback some data
    s = input("Press any key to begin the playback... c to continue")
    if s != 'c':
        print("Playing back the data...")
        data_directory = args.load_from or env.ep_directory
        playback_trajectory(env, data_directory)

    viz_env = VisualizationWrapper(orig_env, indicator_configs=None)
    # generate some data
    while input("Press any key to begin the playback... q to quit") != 'q':
        print("Generating trajectory...")
        generate_trajectory(viz_env, args.load_from or env.ep_directory)
