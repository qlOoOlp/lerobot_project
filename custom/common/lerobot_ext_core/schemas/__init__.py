"""Canonical representation modules — ONE module per representation.

A schema is tied to a *representation*, not to a robot: swapping the robot
reuses the same module, only a representation change needs a new one.

    canonical_ee10.py : EE-Cartesian pose(xyz + rot6d) + gripper, 10D
                        — shared by metaworld/Sawyer, franka, UMI

Every module here exposes the SAME outward contract, so downstream code swaps
representations by changing only the import:

    STATE_DIM, STATE_AXES, ACTION_DIM, ACTION_AXES     # outward contract
    (POSE_DIM / JOINT_DIM ... are representation-internal)

    from custom.common.lerobot_ext_core.schemas import canonical_ee10 as sch
    sch.STATE_DIM

New representation (e.g. joint-space) -> add a sibling module here
(`canonical_joint7.py`); never edit an existing one (open/closed).

Nothing is re-exported on purpose: importing a representation must stay
explicit, so no module can masquerade as "the" schema.
"""
