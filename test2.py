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
gcode_file_path = r"C:\Users\muigims\Desktop\fibo\year3\kinematics\project kine\flatnew.gcode"
layers_of_points = separate_points_by_layers(gcode_file_path)

# Function to calculate Euclidean distance
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Path planning function
def path_planning(layers_of_points):
    num_points = len(layers_of_points)
    distances = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            distances[i][j] = calculate_distance(layers_of_points[i], layers_of_points[j])

    row_ind, col_ind = linear_sum_assignment(distances)
    optimal_order = [layers_of_points[i] for i in col_ind]
    return optimal_order

# Path planning for the points
optimal_path = path_planning(layers_of_points)

# Linear interpolation for trajectory generation
trajectory_points = []
for i in range(len(optimal_path) - 1):
    start_point = optimal_path[i]
    end_point = optimal_path[i + 1]
    num_intermediate_points = 10
    x_interp = np.linspace(start_point[0], end_point[0], num_intermediate_points)
    y_interp = np.linspace(start_point[1], end_point[1], num_intermediate_points)
    z_interp = np.linspace(start_point[2], end_point[2], num_intermediate_points)
    for j in range(num_intermediate_points):
        trajectory_points.append([x_interp[j], y_interp[j], z_interp[j]])

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

# Graph 3: Optimal path and linear trajectory
ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(x_coords, y_coords, z_coords, color='blue', label='Points')
ax3.plot(x_coords, y_coords, z_coords, color='red', linestyle='-', linewidth=1.5, label='Optimal Path')
ax3.plot(trajectory_x_coords, trajectory_y_coords, trajectory_z_coords, color='green', linestyle='-', linewidth=2, label='Linear Trajectory')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Layer')
ax3.set_title('Optimal Path and Linear Trajectory in 3D')
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

# แปลง trajectory_points ให้เป็น numpy array
trajectory_points = np.array(trajectory_points)

# คำนวณความเร็ว (Velocity) โดยใช้ความแตกต่างระหว่างจุดต่อเนื่อง
velocity = np.diff(trajectory_points, axis=0)

# คำนวณความเร่ง (Acceleration) โดยใช้ความแตกต่างของความเร็ว
acceleration = np.diff(velocity, axis=0)

# สร้างแกนเวลา (Time) สำหรับแต่ละจุด
# สมมติว่าเวลาระหว่างจุดเท่ากันและเริ่มต้นที่ t=0
time = np.arange(len(trajectory_points))

# แกนเวลาเพิ่มเติมสำหรับความเร็วและความเร่ง
time_velocity = time[:-1]  # ความเร็วจะมีจุดน้อยกว่าตำแหน่ง 1 จุด
time_acceleration = time[:-2]  # ความเร่งจะมีจุดน้อยกว่าความเร็ว 1 จุด

# สร้างกราฟ
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# กราฟตำแหน่ง (Position)
axs[0].plot(time, trajectory_points[:, 0], label='X Position', color='r')
axs[0].plot(time, trajectory_points[:, 1], label='Y Position', color='g')
axs[0].plot(time, trajectory_points[:, 2], label='Z Position', color='b')
axs[0].set_title('Position (X, Y, Z)')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Position')
axs[0].legend()
axs[0].grid(True)

# กราฟความเร็ว (Velocity)
axs[1].plot(time_velocity, velocity[:, 0], label='X Velocity', color='r')
axs[1].plot(time_velocity, velocity[:, 1], label='Y Velocity', color='g')
axs[1].plot(time_velocity, velocity[:, 2], label='Z Velocity', color='b')
axs[1].set_title('Velocity (X, Y, Z)')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Velocity')
axs[1].legend()
axs[1].grid(True)

# กราฟความเร่ง (Acceleration)
axs[2].plot(time_acceleration, acceleration[:, 0], label='X Acceleration', color='r')
axs[2].plot(time_acceleration, acceleration[:, 1], label='Y Acceleration', color='g')
axs[2].plot(time_acceleration, acceleration[:, 2], label='Z Acceleration', color='b')
axs[2].set_title('Acceleration (X, Y, Z)')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Acceleration')
axs[2].legend()
axs[2].grid(True)

# จัดรูปแบบกราฟ
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
