import argparse

import numpy as np

import genesis as gs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="warning", backend=gs.cpu)
    np.set_printoptions(precision=7, suppress=True)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=200,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            enable_self_collision=True,
            gravity=(0, 0, -0),
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file="/Volumes/Work/Projects/robotics/genesis/genesis/assets/xml/ufactory_lite6/lite6_gripper_wide.xml"),
        # gs.morphs.MJCF(file="/Volumes/Work/Projects/robotics/genesis/genesis/assets/xml/ufactory_lite6/lite6.xml"),
    )

    target_entity = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    ########################## build ##########################
    # n_envs = 5
    n_envs = 1
    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, robot, target_entity, n_envs, args.vis))

    if args.vis:
        scene.viewer.start()

def run_sim(scene, robot, target_entity, n_envs, enable_vis):

    
    # set control gains
    # robot.set_dofs_kp(
    #     np.array([4500, 4500, 3500, 3500, 2000, 2000, 100, 100]),
    # )
    # robot.set_dofs_kv(
    #     np.array([450, 450, 350, 350, 200, 200, 10, 10]),
    # )
    # robot.set_dofs_force_range(
    #     np.array([-87, -87, -87, -87, -12, -12, -100, -100]),
    #     np.array([87, 87, 87, 87, 12, 12, 100, 100]),
    # )

    target_quat = np.tile(np.array([0, 1, 0, 0]), [n_envs, 1])  # pointing downwards
    # center = np.tile(np.array([0.4, -0.2, 0.25]), [n_envs, 1])
    # center = np.tile(np.array([0.2, 0.0, -2.25]), [n_envs, 1])
    # angular_speed = np.random.uniform(-10, 10, n_envs)
    angular_speed = np.random.uniform(-5, 5, n_envs)
    # r = 0.08
    
    ## centered large radius
    center = np.tile(np.array([0.0, 0.0, 0.15]), [n_envs, 1])
    r = 0.16
    # r = 0.5
    # r = 0.05
    
    ee_link = robot.get_link("gripper_body")

    # for i in range(0, 8000):
    i=0
    while True:
        target_pos = np.zeros([n_envs, 3])
        target_pos[:, 0] = center[:, 0] + np.cos(i / 360 * np.pi * angular_speed) * r
        target_pos[:, 1] = center[:, 1] + np.sin(i / 360 * np.pi * angular_speed) * r
        target_pos[:, 2] = center[:, 2]
        # if i==0:
        #     target_q = np.hstack([target_pos, target_quat])
        #     last_q = None
        # else:
        #     last_q = [1, target_q]
        #     target_q = np.hstack([target_pos, target_quat])
        
        target_q = np.hstack([target_pos, target_quat])

        target_entity.set_qpos(target_q)
        
        q = robot.inverse_kinematics(
            link=ee_link,
            # init_qpos=last_q,
            pos=target_pos,
            quat=target_quat,
            rot_mask=[True, True, True],  # for demo purpose: only restrict direction of z-axis
            # rot_mask=[False, False, True],  # for demo purpose: only restrict direction of z-axis
        )

        robot.set_qpos(q)
        # print(q)
        scene.step()
        # if i%100:
            # print(q)
        i+=1
        if abs((i / 360 * np.pi * angular_speed)) > (2*np.pi):
            i=0
            print("reset", q, end=" ")
        
    if enable_vis:
        scene.viewer.stop()

if __name__ == "__main__":
    main()
