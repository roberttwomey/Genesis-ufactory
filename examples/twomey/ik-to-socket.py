import argparse
import numpy as np
import genesis as gs
import socket
import sys

streamJoints = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-t", "--test", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(seed=0, precision="32", logging_level="warning", backend=gs.cpu)
    # gs.init(seed=0, precision="32", logging_level="debug", backend=gs.cpu)
    np.set_printoptions(precision=7, suppress=True)

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
            # gravity=(0, 0, -0),
            gravity=(0, 0, -9.81),       # Proper physics for realistic movements
        ),
    )

    plane = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.MJCF(file="/Volumes/Work/Projects/robotics/genesis/genesis/assets/xml/ufactory_lite6/lite6_gripper_wide.xml")
    )

    target_entity = scene.add_entity(
        gs.morphs.Mesh(file="meshes/axis.obj", scale=0.15),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    scene.build(n_envs=1, env_spacing=(1.0, 1.0))

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, robot, target_entity, args.vis))

    if args.vis:
        scene.viewer.start()

def run_sim(scene, robot, target_entity, enable_vis):

    if streamJoints:
        txsocket = None
        txconn = None
        txadd = None

        txsocket = socket.socket()
        
        print("tx: binding socket")
        # allow socket to reuse address
        txsocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        txport = 12346
        txsocket.bind(('', txport))

        # https://docs.python.org/3/library/socket.html#socket.socket.listen
        txsocket.listen(5) # number of unaccepted connections allow (backlog)

    # target_quat = np.tile(np.array([1, 0, 0, 0]), [1, 1])    
    # target_quat = np.tile(np.array([0, 1, 0, 0]), [1, 1])    
    target_quat = np.tile(np.array([0, 0, 0, 1]), [1, 1])    
    # target_quat = np.tile(np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]), [1, 1])    
    center = np.tile(np.array([0.0, 0.0, 0.25]), [1, 1])
    # r = 0.3
    r = 0.25

    # center = np.tile(np.array([0.0, 0.0, 0.15]), [1, 1])
    # r = 0.16

    ee_link = robot.get_link("gripper_body")

    connected = False

    rot = 0.0
    # speed = 0.1*np.pi/360 # slow
    speed = 0.2*np.pi/360
    dir = 1.0

    while True:
        # center = np.tile(np.array([0.0, 0.0, np.sin(rot)*0.1+0.2]), [1, 1])
        # r = np.sin(rot)*0.10+0.10

        target_pos = np.zeros([1, 3])
        # target_pos[:, 0] = center[:, 0] + np.cos(i / 360 * np.pi * speed ) * r
        # target_pos[:, 1] = center[:, 1] + np.sin(i / 360 * np.pi * speed) * r
        target_pos[:, 0] = center[:, 0] + np.cos(rot) * r
        target_pos[:, 1] = center[:, 1] + np.sin(rot) * r
        target_pos[:, 2] = center[:, 2]

        target_q = np.hstack([target_pos, target_quat])
        target_entity.set_qpos(target_q)

        q = robot.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
            rot_mask=[True, True, True],
        )

        # if robot.check_self_collision(q):
        #     print("Collision detected! Adjusting path...")
        #     # Add collision avoidance logic (e.g., reposition the end effector)
        # else:
        #     robot.set_qpos(q)

        robot.set_qpos(q)
        scene.step()

        if streamJoints:
            if connected:
                try:
                    txconn.send(str(q[0].tolist()).encode())
                except:
                    connected = False
                    print("disconnected.")

                rot += speed * dir
                if abs(rot) > np.pi:
                    dir*=-1.0
                    print("reverse", q, end=" ")
            else: 
                while not connected:
                    print("tx: waiting for connection...")
                    txconn, txaddr = txsocket.accept()
                    connected = True
                
                print("tx: accepted connection from",str(txaddr[0]), ":", str(txaddr[1]))
                txconn.send(str(q[0].tolist()).encode())
                

                # wait for reply
                print("waiting to move to initial position...", end="")
                sys.stdout.flush()
                data = txconn.recv(1024)
                print("done! going", data)
        else:
            rot += speed * dir
            if abs(rot) > np.pi:
                dir*=-1.0
                print("reverse", q, end=" ")

    if streamJoints:
        conn.close()

    if enable_vis:
        scene.viewer.stop()

if __name__ == "__main__":
    main()
