import numpy as np
import argparse
import genesis as gs

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()


    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
            # dt=0.005,
        ),
        show_viewer=True,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    lite6 = scene.add_entity(
        # gs.morphs.MJCF(
        #     file="xml/franka_emika_panda/panda.xml",
        # ),

        gs.morphs.MJCF(
            file="/Volumes/Work/Projects/robotics/genesis/genesis/assets/xml/ufactory_lite6/lite6_gripper_wide.xml",
        ),

    )
    ########################## build ##########################
    scene.build()

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, lite6, args.vis))
    if args.vis:
        scene.viewer.start()

def run_sim(scene, lite6, enable_vis):

    # jnt_names = [
    #     "joint1",
    #     "joint2",
    #     "joint3",
    #     "joint4",
    #     "joint5",
    #     "joint6",
    #     "joint7",
    #     "finger_joint1",
    #     "finger_joint2",
    # ]

    jnt_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "gripper_left_finger",
        "gripper_right_finger",
    ]
        
    dofs_idx = [lite6.get_joint(name).dof_idx_local for name in jnt_names]

    ############ Optional: set control gains ############
    # set positional gains
    lite6.set_dofs_kp(
        # kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 100, 100]),
        dofs_idx_local=dofs_idx,
    )
    # set velocity gains
    lite6.set_dofs_kv(
        # kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        kv=np.array([450, 450, 350, 350, 200, 200, 10, 10]),
        dofs_idx_local=dofs_idx,
    )
    # set force range for safety
    lite6.set_dofs_force_range(
        # lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        # upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        # dofs_idx_local=dofs_idx,
        lower=np.array([-87, -87, -87, -87, -12, -12, -100, -100]),
        upper=np.array([87, 87, 87, 87, 12, 12, 100, 100]),
        dofs_idx_local=dofs_idx,
    )
    # Hard reset
    print("*** Hard Reset")
    for i in range(150):
        if i < 50:
            # lite6.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
            lite6.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
        elif i < 100:
            # lite6.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), dofs_idx)
            lite6.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, 0.04, 0.04]), dofs_idx)
        else:
            # lite6.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), dofs_idx)
            lite6.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0]), dofs_idx)

        scene.step()

    # PD control
    print("*** PD Control")
    for i in range(1250):
        if i == 0:
            lite6.control_dofs_position(
                # np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
                np.array([1, 1, 0, 0, 0, 0, 0.04, 0.04]),
                dofs_idx,
            )
        elif i == 250:
            lite6.control_dofs_position(
                # np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
                np.array([-1, 0.8, 1, -2, 1, 0.5, 0.04, 0.04]),
                dofs_idx,
            )
        elif i == 500:
            lite6.control_dofs_position(
                # np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                dofs_idx,
            )
        elif i == 750:
            # control first dof with velocity, and the rest with position
            lite6.control_dofs_position(
                # np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
                np.array([0, 0, 0, 0, 0, 0, 0, 0])[1:],
                dofs_idx[1:],
            )
            lite6.control_dofs_velocity(
                # np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
                dofs_idx[:1],
                np.array([1.0, 0, 0, 0, 0, 0, 0, 0])[:1],
            )
        elif i == 1000:
            lite6.control_dofs_force(
                # np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                dofs_idx,
            )
        # This is the control force computed based on the given control command
        # If using force control, it's the same as the given control command
        print("control force:", lite6.get_dofs_control_force(dofs_idx))

        # This is the actual force experienced by the dof
        print("internal force:", lite6.get_dofs_force(dofs_idx))

        scene.step()
    
    if enable_vis:
        scene.viewer.stop()

if __name__ == "__main__":
    main()
