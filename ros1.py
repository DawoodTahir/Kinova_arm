#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from sensor_msgs.msg import JointState 
import numpy as np

def dh_transform(a, alpha, d, theta):
    """
    Compute individual transformation matrix using DH parameters.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

import numpy as np

def forward_kinematics(dh_params, joint_angles):
    """
    Compute the forward kinematics for the given DH parameters and joint angles.
    Returns the transformation matrix, position, and orientation separately.
    """
    T = np.eye(4)  # Identity matrix to start transformation
    for i, (a, alpha, d, theta_offset) in enumerate(dh_params):
        theta = joint_angles[i] + theta_offset
        T = np.dot(T, dh_transform(a, alpha, d, theta))
    
    position = T[:3, 3]  # Extract the position from the transformation matrix
    orientation = T[:3, :3]  # Extract the orientation (rotation matrix) from the transformation matrix
    return T, position, orientation


import numpy as np

def rotation_matrix_to_axis_angle(R):
    """Converts a rotation matrix to axis-angle representation."""
    angle = np.arccos((np.trace(R) - 1) / 2)
    if np.isclose(angle, 0):
        return np.zeros(3)  # No rotation
    elif np.isclose(angle, np.pi):
        # Special case for a 180-degree rotation
        # Find the axis of rotation for a 180-degree rotation
        half_trace = (R + np.eye(3)) / 2
        axis = np.array([half_trace[2, 1] - half_trace[1, 2],
                         half_trace[0, 2] - half_trace[2, 0],
                         half_trace[1, 0] - half_trace[0, 1]])
        return axis * np.pi
    else:
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(angle))
        return axis * angle

def inverse_kinematics(dh_params, target_position, target_orientation, joint_angles):
    """
    Numerical inverse kinematics using the Jacobian transpose method for a 7-DOF manipulator.
    """
    q = joint_angles
    max_iterations = 2000
    position_tolerance = 1e-6
    orientation_tolerance = 1e-6

    # Weights for combining position and orientation error
    position_weight = 1.0
    orientation_weight = 1.0

    # Iterative inverse kinematics solver
    for _ in range(max_iterations):
        # Compute current pose
        current_pose, current_position, current_orientation = forward_kinematics(dh_params, q)

        # Position error (3D vector)
        position_error = target_position - current_position

        # Orientation error (3D axis-angle vector)
        rotation_diff = np.dot(current_orientation.T, target_orientation)  # Relative rotation matrix
        orientation_error = rotation_matrix_to_axis_angle(rotation_diff)  # 3D orientation error vector

        # Combine position and orientation error into a 6D error vector
        error = np.concatenate([position_error, orientation_error])

        # Check for convergence
        if np.linalg.norm(position_error) < position_tolerance and np.linalg.norm(orientation_error) < orientation_tolerance:
            print("Converged!")
            break

        # Compute Jacobian numerically (6x7 for a 7-DOF manipulator)
        J = np.zeros((6, len(q)))  # 6 rows (3 for position + 3 for orientation), 7 columns (for 7 joints)
        delta = 1e-6
        for i in range(len(q)):
            q_plus = np.copy(q)
            q_plus[i] += delta
            _, position_plus, orientation_plus = forward_kinematics(dh_params, q_plus)

            # Compute translation Jacobian (Jv) - partial derivatives w.r.t position
            Jv = position_plus - current_position

            # Compute rotational Jacobian (Jw) - partial derivatives w.r.t orientation
            rotation_diff_plus = np.dot(current_orientation.T, orientation_plus)
            Jw = rotation_matrix_to_axis_angle(rotation_diff_plus)  # Use axis-angle representation

            # Assemble Jacobian matrix (6x7)
            J[:, i] = np.concatenate([Jv, Jw])

        # Damping factor for stability (damped least squares)
        damping_factor = 0.01
        J_pinv = np.linalg.pinv(J + damping_factor * np.eye(6, J.shape[1]))  # Apply damping to the Jacobian (6x7)

        # Compute the update for the joint angles
        q_update = np.dot(J_pinv, error)

        # Update joint angles
        q += q_update

        # Apply joint limits (ensure the joint angles stay within their limits)
        q = np.clip(q, -np.pi, np.pi)  # Example of joint limits

    return q



class KortexArmController:
    def __init__(self):
        """
        Initialize the KortexArmController.

        Args:
        - joint_topics: List of topics to publish joint positions (one per joint).
        """
        self.robot_name=rospy.get_param('~robot_name','my_gen3')
        rospy.Subscriber("/my_gen3/joint_states", JointState, self.joint_states_callback)
        
        self.current_joint_positions = None

    def joint_states_callback(self, msg):
        """Callback to get the current joint states."""
        self.current_joint_positions = msg.position

    def move_to_joint_trajectory(self, joint_positions):
        """
        Publish a joint trajectory to move the robot.

        Args:
        - joint_positions: List of joint angles for all joints.
        - time_from_start: Time duration to execute the trajectory (in seconds).
        """
        pub=rospy.Publisher('/my_gen3/gen3_joint_trajectory_controller/command',JointTrajectory,queue_size=10)
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]  # Adjust names if necessary]

        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start = rospy.Duration(2.0)

        trajectory_msg.points.append(point)
        pub.publish(trajectory_msg)
        rospy.loginfo('Joint Trajectory Command Sent')
    # def move_to_joint_positions(self, target_positions):
    #     """
    #     Move the robot to the target joint positions.

    #     Args:
    #     - target_positions: List of target joint angles.
    #     """
    #     if len(target_positions) != len(self.joint_publishers):
    #         rospy.logerr("Target positions do not match the number of joints!")
    #         return

    #     joint_state_msg = JointState()
    #     joint_state_msg.header.stamp = rospy.Time.now()
    #     joint_state_msg.position = target_positions
        
    #     # Now publish the joint state message to the correct topic
    #     self.joint_publishers[0].publish(joint_state_msg)

    def wait_for_current_joint_states(self):
        """Wait until joint states are received."""
        while self.current_joint_positions is None:
            rospy.loginfo("Waiting for joint state messages...")
            rospy.sleep(0.5)

if __name__ == "__main__":
    rospy.init_node("kortex_arm_cartesian_controller")

    controller = KortexArmController()

    # Wait for joint states to be available
    controller.wait_for_current_joint_states()

    # Define DH parameters for your robot
    dh_params = [
    (0.0, np.pi, 0.0, 0),  # Base joint
    (0.0, np.pi / 2, -(0.1564 + 0.1284), 0),  # θ1
    (0.0, np.pi / 2, -(0.0054 + 0.0064), np.pi),  # θ2
    (0.0, np.pi / 2, -(0.2104 + 0.2104), np.pi),  # θ3
    (0.0, np.pi / 2, -(0.0064 + 0.0064), np.pi),  # θ4
    (0.0, np.pi / 2, -(0.2084 + 0.1059), np.pi),  # θ5
    (0.0, np.pi, -(0.1059 + 0.0615), np.pi),  # θ6
    ]



    # Example pick_pose and place_pose (4x4 transformation matrices)
    pick_pose = np.array([
        [1, 0, 0, 0.6],
        [0, 1, 0, 0.3],
        [0, 0, 1, 0.2],
        [0, 0, 0, 1]
    ])

    place_pose = np.array([
        [1, 0, 0, 0.8],
        [0, 1, 0, 0.5],
        [0, 0, 1, 0.4],
        [0, 0, 0, 1]
    ])

    # Extract target position and orientation from the 4x4 matrices
    pick_position = pick_pose[:3, 3]  # [x, y, z] position (translation)
    pick_orientation = pick_pose[:3, :3]  # 3x3 rotation matrix (orientation)

    place_position = place_pose[:3, 3]  # [x, y, z] position (translation)
    place_orientation = place_pose[:3, :3]  # 3x3 rotation matrix (orientation)

    # Print the extracted information
    print("Pick position:", pick_position)
    print("Pick orientation:", pick_orientation)
    print("Place position:", place_position)
    print("Place orientation:", place_orientation)

    # Define the joint angles (for example, zero initial angles)
    joint_angles = [0, 0, 0, 0, 0, 0,0]  # Adjust this according to your robot's current joint state

    # Call the inverse kinematics function
    # Assuming 'dh_params' are already defined correctly for your robot
    pick_angles = inverse_kinematics(dh_params, pick_position, pick_orientation, joint_angles)
    place_angles = inverse_kinematics(dh_params, place_position, place_orientation, pick_angles)

    rospy.loginfo("Moving to pick position...")
    controller.move_to_joint_trajectory(pick_angles)
    rospy.sleep(2)  # Wait for the movement to complete

    rospy.loginfo("Moving to place position...")
    controller.move_to_joint_trajectory(place_angles)
    rospy.sleep(2)  # Wait for the movement to complete

    rospy.loginfo("Pick-and-place operation completed!")
