import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def rotate_vector_by_quaternion(vector, quaternion):
    qw, qx, qy, qz = quaternion
    qx = -qx
    qy = -qy
    qz = -qz
    qq = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return np.dot(qq, vector)


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def joint_torque(action, joint_pos_rel, joint_vel_rel):
    def clip_effort(effort, joint_vel, saturation_effort=23.5, effort_limit=23.5, velocity_limit=30.0):
        max_effort = saturation_effort * (1.0 - joint_vel / velocity_limit)
        max_effort = np.clip(max_effort, min=0.0, max=effort_limit)
        min_effort = saturation_effort * (-1.0 - joint_vel / velocity_limit)
        min_effort = np.clip(min_effort, min=-effort_limit, max=0.0)
        return np.clip(effort, min=min_effort, max=max_effort)

    def compute(target_joint_pos, joint_pos, joint_vel, stiffness=50.0, damping=0.5):
        error_pos = target_joint_pos - joint_pos
        error_vel = -joint_vel
        computed_effort = stiffness * error_pos + damping * error_vel
        return clip_effort(computed_effort, joint_vel)

    return compute(action, joint_pos_rel, joint_vel_rel)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return joint_torque(target_q, q, dq)


def isaac2mujoco(inputs):
    return inputs.reshape([3, 4]).T.flatten()


def mujoco2isaac(inputs):
    return inputs.reshape([4, 3]).T.flatten()


if __name__ == "__main__":
    config = {
        "policy_path": "./policy.pt",
        "xml_path": "./robots/go2/scene.xml",
        "simulation_duration": 3000,
        "simulation_dt": 0.002,
        "control_decimation": 10,
        "kps": 50.0,
        "kds": 0.1,
        "default_angles": [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
        "ang_vel_scale": 1.0,
        "dof_pos_scale": 1.0,
        "dof_vel_scale": 1.0,
        "action_scale": 0.2,
        "cmd_scale": 1.0,
        "num_actions": 12,
        "num_obs": 48,
        "cmd_init": [1.5, 0.0, 0.0]
    }

    policy_path = config["policy_path"]
    xml_path = config["xml_path"]

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]

    viewer = mujoco.viewer.launch_passive(m, d)
    policy = torch.jit.load(policy_path)

    default_angles = np.array(config["default_angles"], dtype=np.float32)
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    cmd = np.array(config["cmd_init"], dtype=np.float32)

    counter = 0
    start = time.time()

    while viewer.is_running() and time.time() - start < config["simulation_duration"]:
        step_start = time.time()

        tau = pd_control(target_dof_pos, d.qpos[7:], config["kps"], np.zeros_like(config["kds"]), d.qvel[6:], config["kds"])
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        counter += 1
        if counter % config["control_decimation"] == 0:
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]

            qj = (qj - default_angles) * config["dof_pos_scale"]
            dqj = dqj * config["dof_vel_scale"]
            gravity_orientation = get_gravity_orientation(quat)
            omega = omega * config["ang_vel_scale"]

            obs[:3] = rotate_vector_by_quaternion(d.qvel[:3], quat)
            obs[3:6] = d.qvel[3:6]
            obs[6:9] = gravity_orientation
            obs[9:12] = cmd
            obs[12:12+num_actions] = mujoco2isaac(qj)
            obs[12+num_actions:12+2*num_actions] = mujoco2isaac(dqj)
            obs[12+2*num_actions:12+3*num_actions] = mujoco2isaac(action)

            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = policy(obs_tensor).detach().numpy().squeeze()
            action = isaac2mujoco(action)
            target_dof_pos = action * config["action_scale"] + default_angles

        viewer.sync()
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
