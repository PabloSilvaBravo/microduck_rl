"""Microduck STATIC grasp: the ground-grasp move, pulled into a statically stable crouch whose open beak hovers 1-2 mm
above the floor (Rémi, 2026-10-26: "starting from the normal grasp position, go a bit lower ... up to one millimetre
from the floor ... we don't want to touch it").

The pose comes from the static study (scripts/static_grasp/sg_search.py, notes/static-grasp): both lips within 2 mm of
the floor, the centre of mass 1.37 cm inside the feet (standing: 1.67 cm), 0.19 N.m worst holding torque, every joint
in its range; it holds for 20 s with constant targets and no policy. What a fixed-target script cannot do is GET there
and back: at the robot's servo gain the half-crouch on the way buckles (sg_stiffness.py). So the policy brings the
duck down and up, and holds the pose in between.

Built on the ground-grasp env (same 61-D obs / 14 actions, scripted mouth, object, DR, timeline: robotd's A-button
pick, 4 s period, hand-over at phase 0.70), with:
  - the mouth opening of the static pose; the object where that pose puts the pads (8.7 cm in front of the feet);
  - in the hold window (phase 0.22-0.36, the mouth closes at 0.30-0.36): a Gaussian pull toward the static pose,
    the two lips toward 1.5 mm, stillness, the centre of mass over the feet;
  - any beak or head floor contact costs, from the first newton (the earlier "new grasp" leaned 2-3 N on its beak);
  - the 90° bisector preference off (the static pose's open beak is at 78°).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ground_grasp_env_cfg import make_microduck_ground_grasp_env_cfg

# notes/static-grasp/poses/grasp_sym_1.json (servo order: left leg 5, neck, head pitch, yaw, roll, right leg 5)
STATIC_POSE = [0.1062, -0.0185, -1.5508, 0.8395, 1.5327, -1.5461, -1.2343, 0.0, 0.0,
               -0.1062, 0.0185, 1.5508, -0.8395, -1.5327]
STATIC_JAW_RAD = 0.386           # 22°: the jaw opening of that pose
OBJECT_NOMINAL_XY = (0.087, 0.0)  # the pads' midpoint in that pose, from the standing root (= the foot sites' midpoint)
HOLD_START, HOLD_END = 0.22, 0.36
LIP_TARGET = 0.0015

STATIC_POSE_WEIGHT = 4.0
LIP_HEIGHT_WEIGHT = 4.0
COM_SUPPORT_WEIGHT = 2.0
HOLD_STILL_WEIGHT = 0.05
BEAK_TOUCH_WEIGHT = -2.0          # per step with the head or jaw on the floor (> 0.05 N)
BEAK_FORCE_WEIGHT = -1.5          # per N, from 0 N (the ground-grasp env starts at 1 N for the head, 3 N for the jaw)


def make_microduck_static_grasp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_ground_grasp_env_cfg(play=play)

    cfg.actions["mouth"].open_rad = STATIC_JAW_RAD
    cfg.events["reset_object"].params["nominal_xy"] = OBJECT_NOMINAL_XY
    cfg.rewards.pop("beak_bisector", None)

    for name in ("head_floor_impact", "jaw_floor_impact"):
        cfg.rewards[name].weight = BEAK_FORCE_WEIGHT
        cfg.rewards[name].params["threshold"] = 0.0
    cfg.rewards["beak_floor_touch"] = RewardTermCfg(
        func=microduck_mdp.beak_floor_touch, weight=BEAK_TOUCH_WEIGHT,
        params={"sensor_names": ("head_floor_contact", "jaw_floor_contact"), "threshold": 0.05},
    )
    cfg.rewards["static_pose"] = RewardTermCfg(
        func=microduck_mdp.static_pose_track, weight=STATIC_POSE_WEIGHT,
        params={"target": STATIC_POSE, "command_name": "twist", "std": 0.25,
                "phase_start": HOLD_START, "phase_end": HOLD_END},
    )
    cfg.rewards["lip_height"] = RewardTermCfg(
        func=microduck_mdp.lip_height_track, weight=LIP_HEIGHT_WEIGHT,
        params={"command_name": "twist", "target": LIP_TARGET, "std": 0.004,
                "phase_start": HOLD_START + 0.02, "phase_end": 0.31},
    )
    cfg.rewards["hold_still"] = RewardTermCfg(
        func=microduck_mdp.hold_stillness, weight=HOLD_STILL_WEIGHT,
        params={"command_name": "twist", "phase_start": HOLD_START + 0.02, "phase_end": 0.30},
    )
    cfg.rewards["com_support"] = RewardTermCfg(
        func=microduck_mdp.com_support_margin, weight=COM_SUPPORT_WEIGHT,
        params={"command_name": "twist", "phase_start": HOLD_START, "phase_end": HOLD_END},
    )
    return cfg
