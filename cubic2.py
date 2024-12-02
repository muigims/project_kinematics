import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
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
    Generate a cubic polynomial trajectory from the given optimal path.

    Parameters:
        optimal_path (list): List of [x, y, z] points.
        num_intermediate_points (int): Number of intermediate points between each pair of optimal points.

    Returns:
        trajectory_points (list): List of interpolated points for the cubic trajectory.
    """
    trajectory_points = []

    for i in range(len(optimal_path) - 1):
        # Define start and end points
        start_point = np.array(optimal_path[i])
        end_point = np.array(optimal_path[i + 1])

        # Boundary conditions
        p0 = start_point
        p1 = end_point
        v0 = np.array([0, 0, 0])  # Initial velocity (assume 0 for simplicity)
        v1 = np.array([0, 0, 0])  # Final velocity (assume 0 for simplicity)
        
        # Time for this segment
        T = 1  # You can scale this as needed

        # Solve for cubic coefficients
        a0 = p0
        a1 = v0
        a2 = (3 / T**2) * (p1 - p0) - (2 / T) * v0 - (1 / T) * v1
        a3 = (-2 / T**3) * (p1 - p0) + (1 / T**2) * (v1 + v0)

        # Generate intermediate points
        for t in np.linspace(0, T, num_intermediate_points):
            position = a0 + a1 * t + a2 * t**2 + a3 * t**3
            trajectory_points.append(position)

    return np.array(trajectory_points)

# Provide the path to your G-code file
gcode_file_path = r"C:\Users\muigims\Desktop\fibo\year3\kinematics\project_6525_6555\project_kinematics\new_square.gcode"  # Replace with your G-code file path
layers_of_points = separate_points_by_layers(gcode_file_path)

# Perform path planning
optimal_path = path_planning(layers_of_points)

# Generate cubic polynomial trajectory
trajectory_points_cubic = generate_cubic_trajectory(optimal_path, num_intermediate_points=10)

# Compute velocity and acceleration
time_points = np.linspace(0, len(trajectory_points_cubic), len(trajectory_points_cubic))
velocity = np.diff(trajectory_points_cubic, axis=0) / np.diff(time_points)[:, None]
acceleration = np.diff(velocity, axis=0) / np.diff(time_points[:-1])[:, None]

# Plot position, velocity, and acceleration
fig, axs = plt.subplots(3, 1, figsize=(12, 15))

# Position plot
axs[0].plot(trajectory_points_cubic[:, 0], label='X Position', color='r')
axs[0].plot(trajectory_points_cubic[:, 1], label='Y Position', color='g')
axs[0].plot(trajectory_points_cubic[:, 2], label='Z Position', color='b')
axs[0].set_title('Position (X, Y, Z)')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Position')
axs[0].legend()
axs[0].grid()

# Velocity plot
axs[1].plot(velocity[:, 0], label='X Velocity', color='r')
axs[1].plot(velocity[:, 1], label='Y Velocity', color='g')
axs[1].plot(velocity[:, 2], label='Z Velocity', color='b')
axs[1].set_title('Velocity (X, Y, Z)')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Velocity')
axs[1].legend()
axs[1].grid()

# Acceleration plot
axs[2].plot(acceleration[:, 0], label='X Acceleration', color='r')
axs[2].plot(acceleration[:, 1], label='Y Acceleration', color='g')
axs[2].plot(acceleration[:, 2], label='Z Acceleration', color='b')
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
ax.plot(trajectory_points_cubic[:, 0], trajectory_points_cubic[:, 1], trajectory_points_cubic[:, 2], label='Cubic Polynomial Trajectory', color='purple', linewidth=2)

# Plot the original points for reference
original_x = [point[0] for point in optimal_path]
original_y = [point[1] for point in optimal_path]
original_z = [point[2] for point in optimal_path]
ax.scatter(original_x, original_y, original_z, color='red', label='Original Points', s=50)

# Set labels and title
ax.set_title('3D Trajectory from Cubic Polynomial', fontsize=14)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.legend()
ax.grid()

# Show the plot
plt.show()


