import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import roboticstoolbox as rtb
from math import pi
from spatialmath import SE3
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linear_sum_assignment
from itertools import permutations

# Function to separate points into layers based on certain commands or comments
def separate_points_by_layers(gcode_file):
    layers_of_points = []
    with open(gcode_file, 'r') as file:
        current_layer = []
        layer_number = 0

        for line in file:
            # Check for commands or comments that indicate a new layer
            if line.startswith(';LAYER'):
                if current_layer:
                    for point in current_layer:
                        layers_of_points.append([point[0], point[1], layer_number])
                    current_layer = []
                    layer_number += 1

            # Extract X and Y coordinates from G0 or G1 commands
            if line.startswith(('G0', 'G1')):
                x = None
                y = None
                if 'X' in line:
                    x = float(line.split('X')[1].split()[0])
                if 'Y' in line:
                    y = float(line.split('Y')[1].split()[0])
                if x is not None or y is not None:
                    current_layer.append([x, y])

        # Append the remaining points from the last layer
        for point in current_layer:
            layers_of_points.append([point[0], point[1], layer_number])

    return layers_of_points

# Provide the path to your G-code file
gcode_file_path = r"C:\Users\muigims\Desktop\fibo\year3\kinematics\project_6525_6555\project_kinematics\new_square.gcode"
layers_of_points = separate_points_by_layers(gcode_file_path)

# Function to calculate Euclidean distance
def calculate_distance(point1, point2):
    return np.sqrt(
        (point1[0] - point2[0])**2 +
        (point1[1] - point2[1])**2 +
        (point1[2] - point2[2])**2
    )

# Path planning function
def path_planning(layers_of_points):
    num_points = len(layers_of_points)
    distances = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            distances[i][j] = calculate_distance(layers_of_points[i], layers_of_points[j])

    row_ind, col_ind = linear_sum_assignment(distances)
    optimal_path = [layers_of_points[i] for i in col_ind]
    return optimal_path

# Path planning for the points
optimal_path = path_planning(layers_of_points)

# Quintic polynomial functions
def quintic_polynomial_coefficients(s0, sf, T):
    delta_s = sf - s0
    a0 = s0
    a1 = 0  # Assuming zero initial velocity
    a2 = 0  # Assuming zero initial acceleration
    a3 = (10 * delta_s) / (T ** 3)
    a4 = (-15 * delta_s) / (T ** 4)
    a5 = (6 * delta_s) / (T ** 5)
    return [a0, a1, a2, a3, a4, a5]

def compute_quintic_position(a_coeffs, t):
    a0, a1, a2, a3, a4, a5 = a_coeffs
    s_t = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5
    return s_t

# Quintic trajectory generation
trajectory_points = []
for i in range(len(optimal_path) - 1):
    start_point = optimal_path[i]
    end_point = optimal_path[i + 1]
    # Compute distance between points
    dist = calculate_distance(start_point, end_point)
    # Set duration proportional to distance (adjust scaling factor as needed)
    T = dist / 10.0  # Adjust as needed for your application
    num_intermediate_points = 10
    time_steps = np.linspace(0, T, num_intermediate_points)
    # Compute coefficients for x, y, z axes
    x_coeffs = quintic_polynomial_coefficients(start_point[0], end_point[0], T)
    y_coeffs = quintic_polynomial_coefficients(start_point[1], end_point[1], T)
    z_coeffs = quintic_polynomial_coefficients(start_point[2], end_point[2], T)
    # Compute positions at each time step
    x_interp = []
    y_interp = []
    z_interp = []
    for t in time_steps:
        x_t = compute_quintic_position(x_coeffs, t)
        y_t = compute_quintic_position(y_coeffs, t)
        z_t = compute_quintic_position(z_coeffs, t)
        x_interp.append(x_t)
        y_interp.append(y_t)
        z_interp.append(z_t)
        trajectory_points.append([x_t, y_t, z_t])

# Extracting coordinates
x_coords = [point[0] for point in optimal_path]
y_coords = [point[1] for point in optimal_path]
z_coords = [point[2] for point in optimal_path]

trajectory_x_coords = [point[0] for point in trajectory_points]
trajectory_y_coords = [point[1] for point in trajectory_points]
trajectory_z_coords = [point[2] for point in trajectory_points]

# Plotting all graphs in one figure
fig = plt.figure(figsize=(18, 12))

# Graph 1: Points classified by layers
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(x_coords, y_coords, z_coords, color='blue', label='Points')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Layer')
ax1.set_title('Points Classified by Layer in 3D')
ax1.legend()

# Graph 2: Optimal path through points
ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(x_coords, y_coords, z_coords, color='blue', label='Points')
ax2.plot(x_coords, y_coords, z_coords, color='red', linestyle='-', linewidth=1.5, label='Optimal Path')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Layer')
ax2.set_title('Optimal Path Through Points in 3D')
ax2.legend()

# Graph 3: Optimal path and quintic trajectory
ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(x_coords, y_coords, z_coords, color='blue', label='Points')
ax3.plot(x_coords, y_coords, z_coords, color='red', linestyle='-', linewidth=1.5, label='Optimal Path')
ax3.plot(trajectory_x_coords, trajectory_y_coords, trajectory_z_coords, color='green', linestyle='-', linewidth=2, label='Quintic Trajectory')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Layer')
ax3.set_title('Optimal Path and Quintic Trajectory in 3D')
ax3.legend()

# Graph 4: Desired end-effector poses
ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(trajectory_x_coords, trajectory_y_coords, trajectory_z_coords, color='red', s=10, label='Desired Poses')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_title('Desired End-Effector Poses')
ax4.legend()

# Adjust layout and show
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Convert trajectory_points to numpy array
trajectory_points = np.array(trajectory_points)

# Calculate velocities and accelerations using gradient
time = np.linspace(0, len(trajectory_points), len(trajectory_points))
velocity = np.gradient(trajectory_points, axis=0)
acceleration = np.gradient(velocity, axis=0)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Position
axs[0].plot(time, trajectory_points[:, 0], label='X Position', color='r')
axs[0].plot(time, trajectory_points[:, 1], label='Y Position', color='g')
axs[0].plot(time, trajectory_points[:, 2], label='Z Position', color='b')
axs[0].set_title('Position (X, Y, Z)')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Position')
axs[0].legend()
axs[0].grid(True)

# Velocity
axs[1].plot(time, velocity[:, 0], label='X Velocity', color='r')
axs[1].plot(time, velocity[:, 1], label='Y Velocity', color='g')
axs[1].plot(time, velocity[:, 2], label='Z Velocity', color='b')
axs[1].set_title('Velocity (X, Y, Z)')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Velocity')
axs[1].legend()
axs[1].grid(True)

# Acceleration
axs[2].plot(time, acceleration[:, 0], label='X Acceleration', color='r')
axs[2].plot(time, acceleration[:, 1], label='Y Acceleration', color='g')
axs[2].plot(time, acceleration[:, 2], label='Z Acceleration', color='b')
axs[2].set_title('Acceleration (X, Y, Z)')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Acceleration')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Assuming trajectory_points is already defined and is a numpy array
trajectory_points = np.array(trajectory_points)  # Ensure trajectory_points is a numpy array

# Create the figure and 3D axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set axis limits (adjust as needed based on your data)
ax.set_xlim(min(trajectory_points[:, 0]), max(trajectory_points[:, 0]))
ax.set_ylim(min(trajectory_points[:, 1]), max(trajectory_points[:, 1]))
ax.set_zlim(min(trajectory_points[:, 2]), max(trajectory_points[:, 2]))

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Trajectory Animation')

# Plot the full trajectory as a static background
ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], color='blue', label='Trajectory')

# Initialize the moving point
point, = ax.plot([], [], [], 'ro', label='Moving Point')

# Animation update function
def update(frame):
    # Update the point's position
    point.set_data([trajectory_points[frame, 0]], [trajectory_points[frame, 1]])
    point.set_3d_properties([trajectory_points[frame, 2]])
    return point,

# Create the animation
ani = FuncAnimation(
    fig, update, frames=len(trajectory_points), interval=100, blit=False
)

# Show the animation
plt.show()

# -------------------------------------------------------------------------------------------------
import roboticstoolbox as rtb
import numpy as np
from math import pi
from spatialmath import SE3

# Robot Parameters
a_1 = 0.145  # Length of the first arm segment (meters)
a_2 = 0.145  # Length of the second arm segment (meters)
d3_min = 0.0   # Minimum Z position of the prismatic joint (meters)
d3_max = 0.2   # Maximum Z position of the prismatic joint (meters)

# Create SCARA robot using MDH parameters
robot = rtb.DHRobot([
    rtb.RevoluteMDH(alpha=0.0, a=a_1, d=0.0, offset=0.0, qlim=[-pi, pi]),
    rtb.RevoluteMDH(alpha=0.0, a=a_2, d=0.0, offset=0.0, qlim=[-pi, pi]),
    rtb.PrismaticMDH(alpha=0.0, a=0.0, theta=0.0, offset=0.0, qlim=[d3_min, d3_max]),
], name='SCARA Robot')

# Extract coordinates from optimal_path and convert to meters
x_coords = [point[0] / 1000.0 for point in optimal_path]
y_coords = [point[1] / 1000.0 for point in optimal_path]
z_coords = [point[2] / 1000.0 for point in optimal_path]

# Generate Workspace
# Create a grid of joint angles
theta1 = np.linspace(-np.pi, np.pi, 300)
theta2 = np.linspace(-np.pi, np.pi, 300)
Theta1, Theta2 = np.meshgrid(theta1, theta2)

# Compute end-effector positions in the XY plane
X = a_1 * np.cos(Theta1) + a_2 * np.cos(Theta1 + Theta2)
Y = a_1 * np.sin(Theta1) + a_2 * np.sin(Theta1 + Theta2)

# Visualization
plt.figure(figsize=(8, 8))
plt.plot(X, Y, '.', markersize=1, label='Workspace')  # Workspace
plt.plot(x_coords, y_coords, c='r', label='Motion Path')  # Path from optimal_path
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title('SCARA Robot Workspace with Motion Path')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

# Inverse Kinematics Function
def calculate_ik(x_coords, y_coords, z_coords, a_1, a_2):
    joint_trajectory = []  # To store joint angles (q1, q2, d3)
    for x, y, z in zip(x_coords, y_coords, z_coords):
        r = np.hypot(x, y)  # Calculate r = sqrt(x^2 + y^2)

        # Calculate cos and sin of q2
        cos_q2 = (r**2 - a_1**2 - a_2**2) / (2 * a_1 * a_2)
        if abs(cos_q2) > 1.0:
            print(f"Target position ({x:.2f}, {y:.2f}, {z:.2f}) is out of reach")
            joint_trajectory.append(None)
            continue
        sin_q2 = np.sqrt(1 - cos_q2**2)

        # Two options for q2
        q2_options = [np.arctan2(sin_q2, cos_q2), np.arctan2(-sin_q2, cos_q2)]

        # Choose appropriate q2
        for q2 in q2_options:
            k1 = a_1 + a_2 * np.cos(q2)
            k2 = a_2 * np.sin(q2)
            q1 = np.arctan2(y, x) - np.arctan2(k2, k1)  # Calculate q1
            d3 = z  # Since the base is at z=0

            q = np.array([q1, q2, d3])

            # Check joint limits
            if robot.islimit(q):
                continue  # Skip if out of limits
            else:
                joint_trajectory.append(q)
                break  # Use the first valid solution
        else:
            joint_trajectory.append(None)  # No valid solution found
    return joint_trajectory

# Call IK function
joint_trajectory = calculate_ik(x_coords, y_coords, z_coords, a_1, a_2)

# Display Results
for i, joint in enumerate(joint_trajectory):
    if joint is not None:
        q1, q2, d3 = joint
        print(f"Point {i+1}:")
        print(f"  q1 = {np.degrees(q1):.2f} degrees")
        print(f"  q2 = {np.degrees(q2):.2f} degrees")
        print(f"  d3 = {d3:.3f} meters")
    else:
        print(f"Point {i+1}: Unreachable")

# Optional: Animate the robot motion using the joint trajectories
# Uncomment the following lines if you have the required visualization libraries

# from roboticstoolbox.backends.swift import Swift
# backend = Swift()
# backend.launch()
# backend.add(robot)
# for q in joint_trajectory:
#     if q is not None:
#         robot.q = q
#         backend.step()
