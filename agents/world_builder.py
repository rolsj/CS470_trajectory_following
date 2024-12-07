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
            <material name="wall">
                <color rgba="0.5 0.5 0.5 1" />
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
            <material name="wall">
                <color rgba="0.5 0.5 0.5 1" />
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
    
    <!-- Fixed joint connecting the two boxes -->
    <joint name="box1_to_box2" type="fixed">
        <parent link="box1_link" />
        <child link="box2_link" />
        <origin xyz="0 0 0" />
    </joint>
</robot>
"""

def build_world(output_dir='../gym_pybullet_drones/assets/world',
                batch=False,
                n=10,
                h1=1.0,
                h2=1.0,
                l=3.0):
    """
    Create a URDF file with two box objects:
    - Box 1 at (0, 0, 0) with height `h1`.
    - Box 2 at (0, 0, l) with height `h2`.

    Parameters:
        filename (str): The name of the URDF file to create.
        batch (bool): Whether it is random batch generating or not
        n (int): Size of batch
        h1 (float): Height of the first box.
        h2 (float): Height of the second box.
        l (float): Z-offset of the second box.
    
    Return:
        world_file_names (string[]): list of output file name with format 'filename_h1_h2_l'
    """
    max_scale_h = 3
    max_scale_l = 3
    world_file_names = []
    N = n if batch else 1
    
    for i in range(N):
        if batch:
            h1 = round(random.random() * max_scale_h, 1)
            h2 = round(random.random() * max_scale_h, 1)
            l = round(random.random() * max_scale_l, 1)
        urdf_content = get_urdf(h1, h2, l)
        
        filename = output_dir.split('/')[-1]
        suffix = f'_{h1}_{h2}_{l}'

        with open(f'{output_dir}{suffix}.urdf', "w") as file:
            file.write(urdf_content)
            
        world_file_names.append(f'{filename}{suffix}')
        
        print(f"URDF file '{filename}{suffix}.urdf' created successfully!")
    
    return world_file_names
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default="../gym_pybullet_drones/assets/world", type=str, metavar='')
    parser.add_argument('--batch', default=False, type=str2bool, metavar='')
    parser.add_argument('--n', default=5, type=int, metavar='')
    parser.add_argument('--h1', default=1.0, type=float, metavar='')
    parser.add_argument('--h2', default=1.0, type=float, metavar='')
    parser.add_argument('--l', default=2.0, type=float, metavar='')
    ARGS = parser.parse_args()

    build_world(**vars(ARGS))
    #build_world("world.urdf", H1=2.0, H2=1.5, L=3.0)
