"""Static grasp: the ground-grasp env pulled into the static pose (reward signs, pose in range, registration)."""

import numpy as np

import mjlab_microduck.tasks  # noqa: F401  (registry first)
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot.ground_grasp import ground_grasp_joint_limits
from mjlab_microduck.tasks import mdp as M
from mjlab_microduck.tasks.microduck_static_grasp_env_cfg import (
    OBJECT_NOMINAL_XY,
    STATIC_JAW_RAD,
    STATIC_POSE,
    make_microduck_static_grasp_env_cfg,
)


def test_registered():
    cfg = load_env_cfg("Mjlab-StaticGrasp-FromPick-MicroDuck")
    assert "static_pose" in cfg.rewards


def test_reward_signs_and_terms():
    r = make_microduck_static_grasp_env_cfg().rewards
    assert "beak_bisector" not in r
    # positive costs take negative weights, self-negating penalties positive ones (AGENTS.md sign rule)
    assert r["beak_floor_touch"].func is M.beak_floor_touch and r["beak_floor_touch"].weight < 0
    for name in ("head_floor_impact", "jaw_floor_impact"):
        assert r[name].weight < 0 and r[name].params["threshold"] == 0.0
    assert r["hold_still"].func is M.hold_stillness and r["hold_still"].weight > 0
    for name in ("static_pose", "lip_height", "com_support", "grasp_held"):
        assert r[name].weight > 0


def test_mouth_and_object_follow_the_static_pose():
    cfg = make_microduck_static_grasp_env_cfg()
    assert cfg.actions["mouth"].open_rad == STATIC_JAW_RAD
    assert tuple(cfg.events["reset_object"].params["nominal_xy"]) == OBJECT_NOMINAL_XY


def test_static_pose_inside_the_joint_ranges():
    lim = np.array(list(ground_grasp_joint_limits().values()))
    q = np.array(STATIC_POSE)
    assert lim.shape == (14, 2)
    assert np.all(q >= lim[:, 0]) and np.all(q <= lim[:, 1])
