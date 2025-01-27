import numpy as np

import genesis as gs

def main():
        
    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug")

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=False,
            enable_collision=False,
        ),
    )

    ########################## entities ##########################

    scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        # gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        gs.morphs.MJCF(file="/Volumes/Work/Projects/robotics/genesis/genesis/assets/xml/ufactory_lite6/lite6_gripper_wide.xml",),
    )

    # two target links for visualization
    target_left = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.1,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )
    target_right = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.1,
        ),
        surface=gs.surfaces.Default(color=(0.5, 1.0, 0.5, 1)),
    )

    ########################## build ##########################
    scene.build()

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, robot, target_left, target_right, True))
    # if args.vis:
    #    scene.viewer.start()
    scene.viewer.start()

def run_sim(scene, robot, target_left, target_right, enable_vis):

    target_quat = np.array([0, 1, 0, 0])
    # center = np.array([0.4, -0.2, 0.25])
    # r = 0.1
    center = np.array([0.3, 0.2, 0.25])
    r = 0.08

    left_finger = robot.get_link("gripper_left_finger")
    right_finger = robot.get_link("gripper_right_finger")

    for i in range(0, 2000):
        target_pos_left = center + np.array([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * r
        target_pos_right = target_pos_left + np.array([0.0, -0.04, 0])

        target_left.set_qpos(np.concatenate([target_pos_left, target_quat]))
        target_right.set_qpos(np.concatenate([target_pos_right, target_quat]))

        q = robot.inverse_kinematics_multilink(
            links=[left_finger, right_finger],
            poss=[target_pos_left, target_pos_right],
            quats=[target_quat, target_quat],
            rot_mask=[False, False, True],  # only restrict direction of z-axis
        )

        # Note that this IK is for visualization purposes, so here we do not call scene.step(), but only update the state and the visualizer
        # In actual control applications, you should instead use robot.control_dofs_position() and scene.step()
        robot.set_dofs_position(q)
        scene.visualizer.update()
    if enable_vis:
        scene.viewer.stop()

if __name__ == "__main__":
    main()