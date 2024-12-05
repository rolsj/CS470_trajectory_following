def build_world(filename, H1, H2, L):
    """
    Create a URDF file with two box objects:
    - Box 1 at (0, 0, 0) with height `H1`.
    - Box 2 at (0, 0, L) with height `H2`.

    Parameters:
        filename (str): The name of the URDF file to create.
        H1 (float): Height of the first box.
        H2 (float): Height of the second box.
        L (float): Z-offset of the second box.
    """
    urdf_content = f"""<?xml version="1.0"?>
<robot name="two_boxes">
    <!-- First box -->
    <link name="box1_link">
        <visual>
            <geometry>
                <box size="1 1 {H1}" />
            </geometry>
            <origin xyz="0 0 {H1 / 2}" />
            <material name="blue">
                <color rgba="0 0 1 1" />
            </material>
        </visual>
        <collision>
            <geometry>
                <box size="1 1 {H1}" />
            </geometry>
            <origin xyz="0 0 {H1 / 2}" />
        </collision>
        <inertial>
            <mass value="1.0" />
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1" />
            <origin xyz="0 0 {H1 / 2}" />
        </inertial>
    </link>

    <!-- Second box -->
    <link name="box2_link">
        <visual>
            <geometry>
                <box size="1 1 {H2}" />
            </geometry>
            <origin xyz="0 0 {L + H2 / 2}" />
            <material name="red">
                <color rgba="1 0 0 1" />
            </material>
        </visual>
        <collision>
            <geometry>
                <box size="1 1 {H2}" />
            </geometry>
            <origin xyz="0 0 {L + H2 / 2}" />
        </collision>
        <inertial>
            <mass value="1.0" />
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1" />
            <origin xyz="0 0 {L + H2 / 2}" />
        </inertial>
    </link>
</robot>
"""
    with open(filename, "w") as file:
        file.write(urdf_content)
    print(f"URDF file '{filename}' created successfully!")

# Example usage
create_two_box_urdf("world.urdf", H1=2.0, H2=1.5, L=3.0)
