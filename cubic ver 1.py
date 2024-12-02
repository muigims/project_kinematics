import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment

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

# Function to generate cubic trajectory
def generate_cubic_trajectory(optimal_path, num_intermediate_points=10):
    """
    Generate a cubic trajectory from the given optimal path.

    Parameters:
        optimal_path (list): List of [x, y, z] points.
        num_intermediate_points (int): Number of intermediate points between each pair of optimal points.

    Returns:
        trajectory_points (list): List of interpolated points for the cubic trajectory.
    """
    # Separate x, y, and z coordinates from the optimal path
    x_coords = [point[0] for point in optimal_path]
    y_coords = [point[1] for point in optimal_path]
    z_coords = [point[2] for point in optimal_path]

    # Generate time array (t) for the original points
    t = np.arange(len(optimal_path))

    # Create cubic splines for x, y, and z
    cubic_x = CubicSpline(t, x_coords)
    cubic_y = CubicSpline(t, y_coords)
    cubic_z = CubicSpline(t, z_coords)

    # Generate new time array for the interpolated trajectory
    t_new = np.linspace(0, len(optimal_path) - 1, (len(optimal_path) - 1) * num_intermediate_points)

    # Interpolate x, y, z using the cubic splines
    x_interp = cubic_x(t_new)
    y_interp = cubic_y(t_new)
    z_interp = cubic_z(t_new)

    # Combine interpolated points into a trajectory
    trajectory_points = np.vstack((x_interp, y_interp, z_interp)).T
    return trajectory_points, t_new, cubic_x, cubic_y, cubic_z

# Provide the path to your G-code file
gcode_file_path = r"C:\Users\muigims\Desktop\fibo\year3\kinematics\project_6525_6555\project_kinematics\new_square.gcode"  # Replace with your G-code file path
layers_of_points = separate_points_by_layers(gcode_file_path)

# Perform path planning
optimal_path = path_planning(layers_of_points)

# Generate cubic trajectory
trajectory_points_cubic, t_new, cubic_x, cubic_y, cubic_z = generate_cubic_trajectory(optimal_path, num_intermediate_points=10)

# Compute velocity and acceleration
velocity_x = cubic_x(t_new, 1)  # First derivative for velocity
velocity_y = cubic_y(t_new, 1)
velocity_z = cubic_z(t_new, 1)

acceleration_x = cubic_x(t_new, 2)  # Second derivative for acceleration
acceleration_y = cubic_y(t_new, 2)
acceleration_z = cubic_z(t_new, 2)

# Plot position, velocity, and acceleration
fig, axs = plt.subplots(3, 1, figsize=(12, 15))

# Position plot
axs[0].plot(t_new, trajectory_points_cubic[:, 0], label='X Position', color='r')
axs[0].plot(t_new, trajectory_points_cubic[:, 1], label='Y Position', color='g')
axs[0].plot(t_new, trajectory_points_cubic[:, 2], label='Z Position', color='b')
axs[0].set_title('Position (X, Y, Z)')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Position')
axs[0].legend()
axs[0].grid()

# Velocity plot
axs[1].plot(t_new, velocity_x, label='X Velocity', color='r')
axs[1].plot(t_new, velocity_y, label='Y Velocity', color='g')
axs[1].plot(t_new, velocity_z, label='Z Velocity', color='b')
axs[1].set_title('Velocity (X, Y, Z)')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Velocity')
axs[1].legend()
axs[1].grid()

# Acceleration plot
axs[2].plot(t_new, acceleration_x, label='X Acceleration', color='r')
axs[2].plot(t_new, acceleration_y, label='Y Acceleration', color='g')
axs[2].plot(t_new, acceleration_z, label='Z Acceleration', color='b')
axs[2].set_title('Acceleration (X, Y, Z)')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Acceleration')
axs[2].legend()
axs[2].grid()

# Adjust layout
plt.tight_layout()
plt.show()


# Plot 3D trajectory
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(trajectory_points_cubic[:, 0], trajectory_points_cubic[:, 1], trajectory_points_cubic[:, 2], label='Cubic Trajectory', color='purple', linewidth=2)

# Plot the original points for reference
original_x = [point[0] for point in optimal_path]
original_y = [point[1] for point in optimal_path]
original_z = [point[2] for point in optimal_path]
ax.scatter(original_x, original_y, original_z, color='red', label='Original Points', s=50)

# Set labels and title
ax.set_title('3D Trajectory from Cubic Interpolation', fontsize=14)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.legend()
ax.grid()

# Show the plot
plt.show()
