import argparse
import random

from gym_pybullet_drones.utils.utils import str2bool

def get_urdf(h1, h2, l):
    return f"""<?xml version="1.0"?>
<robot name="two_boxes">
    <!-- First box -->
    <link name="box1_link">
        <visual>
            <geometry>
                <box size="0.1 1 {h1}" />
            </geometry>
            <origin xyz="0 0 {h1 / 2}" />
            <material name="blue">
                <color rgba="0 0 1 1" />
            </material>
        </visual>
        <collision>
            <geometry>
                <box size="0.1 1 {h1}" />
            </geometry>
            <origin xyz="0 0 {h1 / 2}" />
        </collision>
        <inertial>
            <mass value="1.0" />
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1" />
            <origin xyz="0 0 {h1 / 2}" />
        </inertial>
    </link>

    <!-- Second box -->
    <link name="box2_link">
        <visual>
            <geometry>
                <box size="0.1 1 {h2}" />
            </geometry>
            <origin xyz="{l} 0 {h2 / 2}" />
            <material name="red">
                <color rgba="1 0 0 1" />
            </material>
        </visual>
        <collision>
            <geometry>
                <box size="0.1 1 {h2}" />
            </geometry>
            <origin xyz="{l} 0 {h2 / 2}" />
        </collision>
        <inertial>
            <mass value="1.0" />
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1" />
            <origin xyz="{l} 0 {h2 / 2}" />
        </inertial>
    </link>
</robot>
"""

def build_world(filename, batch, n, h1, h2, l):
    """
    Create a URDF file with two box objects:
    - Box 1 at (0, 0, 0) with height `h1`.
    - Box 2 at (0, 0, l) with height `h2`.

    Parameters:
        filename (str): The name of the URDF file to create.
        batch (bool): Whether it is random batch generating or not
        h1 (float): Height of the first box.
        h2 (float): Height of the second box.
        l (float): Z-offset of the second box.
    """
    max_scale_h = 3
    max_scale_l = 3

    if batch:
        for i in range(n):
            h1 = round(random.random() * max_scale_h, 1)
            h2 = round(random.random() * max_scale_h, 1)
            l = round(random.random() * max_scale_l, 1)
            urdf_content = get_urdf(h1, h2, l)

            with open(f'{filename}_{h1}_{h2}_{l}.urdf', "w") as file:
                file.write(urdf_content)
    else:
        urdf_content = get_urdf(h1, h2, l)
        with open(f'{filename}.urdf', "w") as file:
            file.write(urdf_content)
    print(f"URDF file '{filename}' created successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default="world", type=str, help='directory for the output .urdf file', metavar='')
    parser.add_argument('--batch', default=False, type=str2bool, help='whether to use randomized batch generation', metavar='')
    parser.add_argument('--n', default=5, type=int, help='the size of batch when batch=true', metavar='')
    parser.add_argument('--h1', default=1.0, type=float, help='h1, if batch=fasle', metavar='')
    parser.add_argument('--h2', default=1.0, type=float, help='h2, if batch=fasle', metavar='')
    parser.add_argument('--l', default=3.0, type=float, help='l, if batch=fasle', metavar='')
    ARGS = parser.parse_args()

    build_world(**vars(ARGS))
    #build_world("world.urdf", H1=2.0, H2=1.5, L=3.0)
