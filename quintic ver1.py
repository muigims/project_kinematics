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

# Quintic polynomial trajectory function
def quintic_trajectory(t, p0, p1, v0, v1, a0, a1):
    """
    Compute the quintic polynomial coefficients for trajectory.

    Parameters:
        t (float): Time duration of trajectory.
        p0, p1: Start and end positions.
        v0, v1: Start and end velocities.
        a0, a1: Start and end accelerations.

    Returns:
        Function for position, velocity, and acceleration.
    """
    A = np.array([
        [0, 0, 0, 0, 0, 1],
        [t**5, t**4, t**3, t**2, t, 1],
        [0, 0, 0, 0, 1, 0],
        [5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0],
        [0, 0, 0, 2, 0, 0],
        [20*t**3, 12*t**2, 6*t, 2, 0, 0]
    ])
    b = np.array([p0, p1, v0, v1, a0, a1])
    coeffs = np.linalg.solve(A, b)
    return coeffs

# Generate trajectory using quintic polynomial
def generate_quintic_trajectory(optimal_path, num_intermediate_points=10, total_time=1.0):
    trajectory_points = []
    velocities = []
    accelerations = []

    for i in range(len(optimal_path) - 1):
        p0 = np.array(optimal_path[i])
        p1 = np.array(optimal_path[i + 1])
        v0 = v1 = np.zeros_like(p0)  # Start and end velocities
        a0 = a1 = np.zeros_like(p0)  # Start and end accelerations
        t = np.linspace(0, total_time, num_intermediate_points)

        trajectory_segment = []
        velocity_segment = []
        acceleration_segment = []

        for dim in range(len(p0)):
            coeffs = quintic_trajectory(total_time, p0[dim], p1[dim], v0[dim], v1[dim], a0[dim], a1[dim])
            position = np.polyval(coeffs, t)
            velocity = np.polyval(np.polyder(coeffs, 1), t)
            acceleration = np.polyval(np.polyder(coeffs, 2), t)

            trajectory_segment.append(position)
            velocity_segment.append(velocity)
            acceleration_segment.append(acceleration)

        trajectory_points.append(np.array(trajectory_segment).T)
        velocities.append(np.array(velocity_segment).T)
        accelerations.append(np.array(acceleration_segment).T)

    trajectory_points = np.vstack(trajectory_points)
    velocities = np.vstack(velocities)
    accelerations = np.vstack(accelerations)

    return trajectory_points, velocities, accelerations, t

# Provide the path to your G-code file
gcode_file_path = r"C:\Users\muigims\Desktop\fibo\year3\kinematics\project_6525_6555\project_kinematics\new_square.gcode"  # Replace with your G-code file path
layers_of_points = separate_points_by_layers(gcode_file_path)

# Perform path planning
optimal_path = path_planning(layers_of_points)

# Generate quintic trajectory
trajectory_points, velocities, accelerations, t = generate_quintic_trajectory(optimal_path, num_intermediate_points=20)

# Plot position, velocity, and acceleration
fig, axs = plt.subplots(3, 1, figsize=(12, 15))

# Position plot
axs[0].plot(trajectory_points[:, 0], label='X Position', color='r')
axs[0].plot(trajectory_points[:, 1], label='Y Position', color='g')
axs[0].plot(trajectory_points[:, 2], label='Z Position', color='b')
axs[0].set_title('Position (X, Y, Z)')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Position')
axs[0].legend()
axs[0].grid()

# Velocity plot
axs[1].plot(velocities[:, 0], label='X Velocity', color='r')
axs[1].plot(velocities[:, 1], label='Y Velocity', color='g')
axs[1].plot(velocities[:, 2], label='Z Velocity', color='b')
axs[1].set_title('Velocity (X, Y, Z)')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Velocity')
axs[1].legend()
axs[1].grid()

# Acceleration plot
axs[2].plot(accelerations[:, 0], label='X Acceleration', color='r')
axs[2].plot(accelerations[:, 1], label='Y Acceleration', color='g')
axs[2].plot(accelerations[:, 2], label='Z Acceleration', color='b')
axs[2].set_title('Acceleration (X, Y, Z)')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Acceleration')
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()


# 3D Trajectory Plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the quintic trajectory
ax.plot(
    trajectory_points[:, 0],
    trajectory_points[:, 1],
    trajectory_points[:, 2],
    label='Quintic Trajectory',
    color='purple',
    linewidth=2
)

# Plot original points for reference
original_x = [point[0] for point in optimal_path]
original_y = [point[1] for point in optimal_path]
original_z = [point[2] for point in optimal_path]

ax.scatter(
    original_x,
    original_y,
    original_z,
    color='red',
    label='Original Points',
    s=50
)

# Label the axes
ax.set_title('3D Quintic Trajectory', fontsize=14)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')

# Add legend and grid
ax.legend()
ax.grid()

# Show the plot
plt.show()