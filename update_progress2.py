import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import roboticstoolbox as rtb
from math import pi
from spatialmath import SE3
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linear_sum_assignment
from itertools import permutations
from scipy.interpolate import CubicSpline

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



# Quintic Trajectory Generation Function
def quintic_trajectory(start, end, T, dt=0.01):
    """
    Generates a quintic trajectory between start and end points.
    
    Parameters:
    - start: Starting point [x0, y0, z0]
    - end: Ending point [xf, yf, zf]
    - T: Total time to move from start to end
    - dt: Sampling time
    
    Returns:
    - trajectory: Array of points along the quintic trajectory
    """
    # Boundary conditions
    # Position
    x0, y0, z0 = start
    xf, yf, zf = end
    # Velocities and accelerations
    v0 = [0, 0, 0]
    vf = [0, 0, 0]
    a0 = [0, 0, 0]
    af = [0, 0, 0]

    # Time vector
    t = np.linspace(0, T, int(T/dt) + 1)

    # Initialize trajectory
    traj = np.zeros((len(t), 3))

    for i, axis in enumerate(['x', 'y', 'z']):
        # Coefficients for quintic polynomial
        # Using boundary conditions: p0, pf, v0, vf, a0, af
        # The quintic polynomial is: p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        # We solve for coefficients based on boundary conditions
        A = np.array([
            [1, 0,    0,     0,      0,       0],
            [0, 1,    0,     0,      0,       0],
            [0, 0,    2,     0,      0,       0],
            [1, T, T**2, T**3,  T**4,   T**5],
            [0, 1,  2*T,  3*T**2,4*T**3, 5*T**4],
            [0, 0,    2,  6*T,   12*T**2,20*T**3]
        ])
        b = np.array([
            start[i],
            v0[i],
            a0[i],
            end[i],
            vf[i],
            af[i]
        ])
        coeffs = np.linalg.solve(A, b)
        # Compute position for each time step
        traj[:, i] = coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5
    
    return traj

    
# Generate Quintic Trajectory for all segments
def generate_quintic_trajectory(optimal_path, T_segment=1.0, dt=0.01):
    """
    Generates a complete quintic trajectory through all waypoints.
    
    Parameters:
    - optimal_path: List of waypoints [[x, y, z], ...]
    - T_segment: Time allocated for each segment
    - dt: Sampling time
    
    Returns:
    - full_traj: Array of all trajectory points
    """
    full_traj = []
    num_waypoints = len(optimal_path)
    for i in range(num_waypoints - 1):
        start = optimal_path[i]
        end = optimal_path[i + 1]
        traj_segment = quintic_trajectory(start, end, T_segment, dt)
        if i > 0:
            # Avoid duplicating points at the waypoints
            traj_segment = traj_segment[1:]
        full_traj.append(traj_segment)
    full_traj = np.vstack(full_traj)
    return full_traj

# Generate the quintic trajectory with sampling time = 0.01
sampling_time = 0.01  # seconds
time_per_segment = 1.0  # Adjust as needed
trajectory_points = generate_quintic_trajectory(optimal_path, T_segment=time_per_segment, dt=sampling_time)




# Extracting coordinates
x_coords = [point[0] for point in optimal_path]
y_coords = [point[1] for point in optimal_path]
z_coords = [point[2] for point in optimal_path]

trajectory_x_coords = trajectory_points[:, 0]
trajectory_y_coords = trajectory_points[:, 1]
trajectory_z_coords = trajectory_points[:, 2]


trajectory_x_coords = trajectory_x_coords
trajectory_y_coords = trajectory_y_coords

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
# ax3.plot(x_coords, y_coords, z_coords, color='red', linestyle='-', linewidth=1.5, label='Optimal Path')
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

# Velocity and Acceleration Calculations for Quintic Trajectory
# Calculate velocity using numerical differentiation
velocity = np.diff(trajectory_points, axis=0) / sampling_time

# Calculate acceleration using numerical differentiation of velocity
acceleration = np.diff(velocity, axis=0) / sampling_time

# Create time vectors
time_total = trajectory_points.shape[0] * sampling_time
time = np.arange(0, trajectory_points.shape[0]) * sampling_time
time_velocity = time[:-1]
time_acceleration = time[:-2]


# Plot Position, Velocity, and Acceleration
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Position
axs[0].plot(time, trajectory_points[:, 0], label='X Position', color='r')
axs[0].plot(time, trajectory_points[:, 1], label='Y Position', color='g')
axs[0].plot(time, trajectory_points[:, 2], label='Z Position', color='b')
axs[0].set_title('Position (X, Y, Z)')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Position')
axs[0].legend()
axs[0].grid(True)

# Velocity
axs[1].plot(time_velocity, velocity[:, 0], label='X Velocity', color='r')
axs[1].plot(time_velocity, velocity[:, 1], label='Y Velocity', color='g')
axs[1].plot(time_velocity, velocity[:, 2], label='Z Velocity', color='b')
axs[1].set_title('Velocity (X, Y, Z)')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Velocity')
axs[1].legend()
axs[1].grid(True)

# Acceleration
axs[2].plot(time_acceleration, acceleration[:, 0], label='X Acceleration', color='r')
axs[2].plot(time_acceleration, acceleration[:, 1], label='Y Acceleration', color='g')
axs[2].plot(time_acceleration, acceleration[:, 2], label='Z Acceleration', color='b')
axs[2].set_title('Acceleration (X, Y, Z)')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Acceleration')
axs[2].legend()
axs[2].grid(True)

# Adjust layout and show
plt.tight_layout()
plt.show()


# Animation of Quintic Trajectory
from matplotlib.animation import FuncAnimation

# Create the figure and 3D axes
fig_anim = plt.figure(figsize=(10, 8))
ax_anim = fig_anim.add_subplot(111, projection='3d')

# Set axis limits based on trajectory
ax_anim.set_xlim(np.min(trajectory_points[:, 0]) - 1, np.max(trajectory_points[:, 0]) + 1)
ax_anim.set_ylim(np.min(trajectory_points[:, 1]) - 1, np.max(trajectory_points[:, 1]) + 1)
ax_anim.set_zlim(np.min(trajectory_points[:, 2]) - 1, np.max(trajectory_points[:, 2]) + 1)

# Set labels
ax_anim.set_xlabel('X')
ax_anim.set_ylabel('Y')
ax_anim.set_zlabel('Z')
ax_anim.set_title('3D Quintic Trajectory Animation')

# Plot the full trajectory as a static background
ax_anim.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], color='blue', label='Quintic Trajectory')

# Initialize the moving point
point, = ax_anim.plot([], [], [], 'ro', label='Moving Point')

# Animation update function
def update(frame):
    # Update the point's position
    point.set_data([trajectory_points[frame, 0]], [trajectory_points[frame, 1]])
    point.set_3d_properties([trajectory_points[frame, 2]])
    return point,

# Create the animation
ani = FuncAnimation(
    fig_anim, update, frames=len(trajectory_points), interval=10, blit=False
)

# Show the animation
plt.legend()
plt.show()

import pandas as pd
import numpy as np

df_position = pd.DataFrame({
    'Time[s]': time,
    'X_Position': trajectory_points[:, 0],
    'Y_Position': trajectory_points[:, 1],
    'Z_Position': trajectory_points[:, 2]
})

df_position.to_csv('position.csv', index=False)




# import roboticstoolbox as rtb
# import numpy as np
# import matplotlib.pyplot as plt

# # ---- Robot Parameters ----
# # ความยาวของแขนหุ่นยนต์ (เมตร)
# # a_1 = 0.145      # ความยาวของแขนแรก (100 มม.)
# # a_2 = 0.145      # ความยาวของแขนที่สอง (100 มม.)
# d3_min = 0.0   # ตำแหน่ง Z ต่ำสุดของข้อต่อ Prismatic (0 มม.)
# d3_max = 50   # ตำแหน่ง Z สูงสุดของข้อต่อ Prismatic (100 มม.)

# # a_1 = 0.320      # ความยาวของแขนแรก (100 มม.)
# # a_2 = 0.320      # ความยาวของแขนที่สอง (100 มม.)

# # สร้างหุ่นยนต์ SCARA ด้วย MDH parameters
# robot = rtb.DHRobot([
#     rtb.RevoluteMDH(alpha=0.0, a=0, d=320, offset=0.0, qlim=[-np.pi, np.pi]),
#     rtb.RevoluteMDH(alpha=0.0, a=320, d=0.0, offset=0.0, qlim=[-np.pi, np.pi]),
#     rtb.PrismaticMDH(alpha=180, a=0.0, theta=0.0, offset=0.0, qlim=[d3_min, d3_max]),
# ], name='SCARA Robot')

# # ---- Generate Workspace ----
# # สร้างกริดของมุมข้อต่อ
# theta1 = np.linspace(np.pi, 0, 300)
# theta2 = np.linspace(30, 0, 300)
# Theta1, Theta2 = np.meshgrid(theta1, theta2)

# # คำนวณตำแหน่งปลายแขนหุ่นยนต์ในระนาบ XY
# X = 320 * np.cos(Theta1) + 320 * np.cos(Theta1 + Theta2)
# Y = 320 * np.sin(Theta1) + 320 * np.sin(Theta1 + Theta2)

# # ---- Use Coordinates from optimal_path ----
# # สมมติว่า optimal_path มีอยู่แล้ว และเป็นลิสต์ของจุด [(x1, y1, z1), (x2, y2, z2), ...]

# # แยกพิกัด x, y, z
# x_coords = [point[0] for point in optimal_path]
# y_coords = [point[1] for point in optimal_path]
# z_coords = [point[2] for point in optimal_path]

# x_coords = np.array(x_coords)
# y_coords = np.array(y_coords)
# z_coords = np.array(z_coords) 
 
# # ---- Visualization ----
# plt.figure(figsize=(8, 8))
# plt.plot(X, Y, '.', markersize=1, label='workspace')  # พื้นที่การทำงาน
# plt.plot(x_coords, y_coords, c='r', label='trajectory')  # เส้นทางจาก optimal_path
# plt.xlabel('X (meter)')
# plt.ylabel('Y (mteter)')
# plt.title('trajectory on workspace')
# plt.axis('equal')
# plt.grid(True)
# plt.legend()
# plt.show()







import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# กำหนดความยาวของแขน
L1 = 320  # mm
L2 = 320  # mm


trajectory_x_coords = trajectory_points[:, 0]
trajectory_y_coords = trajectory_points[:, 1]
trajectory_z_coords = trajectory_points[:, 2]

trajectory_x_coords = trajectory_x_coords + 320
trajectory_y_coords = trajectory_y_coords + 320


# ฟังก์ชันสำหรับคำนวณ inverse kinematics
def inverse_kinematics(x, y, z):
    r_squared = x**2 + y**2
    cos_q2 = (r_squared - L1**2 - L2**2) / (2 * L1 * L2)
    
    if cos_q2 < -1 or cos_q2 > 1:
        return None, None, None
    
    q2 = np.arccos(cos_q2)
    k1 = L1 + L2 * np.cos(q2)
    k2 = L2 * np.sin(q2)
    q1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    q3 = z
    
    return np.degrees(q1), np.degrees(q2), q3

# สร้าง lists สำหรับเก็บค่า q1, q2, q3
q1_values = []
q2_values = []
q3_values = []

# คำนวณค่า q1, q2, q3
for x, y, z in zip(trajectory_x_coords, trajectory_y_coords, trajectory_z_coords):
    q1, q2, q3 = inverse_kinematics(x, y, z)
    if q1 is not None:
        q1_values.append(q1)
        q2_values.append(q2)
        q3_values.append(q3)

# แสดงผลลัพธ์
print(f"Generated {len(q1_values)} trajectory points at 0.01-second intervals.")

# # พล็อต trajectory 3D
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# ax.plot3D(trajectory_x_coords, trajectory_y_coords, trajectory_z_coords, 'blue', label='Trajectory Path (Interpolated)')
# ax.scatter(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], c='red', label='Original Points')

# ax.set_xlabel('X (mm)')
# ax.set_ylabel('Y (mm)')
# ax.set_zlabel('Z (mm)')
# ax.set_title('Interpolated Trajectory Path and End Effector Positions')

# plt.legend()
# plt.show()

# พล็อตมุมข้อต่อ
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Plot q1
ax1.plot(np.arange(len(q1_values)) * 0.01, q1_values, marker='o', color='b', label='q1 (deg)', markersize=1)
ax1.set_title('Joint Angle q1')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('q1 (degrees)')
ax1.legend()

# Plot q2
ax2.plot(np.arange(len(q2_values)) * 0.01, q2_values, marker='o', color='g', label='q2 (deg)', markersize=1)
ax2.set_title('Joint Angle q2')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('q2 (degrees)')
ax2.legend()

# Plot q3
ax3.plot(np.arange(len(q3_values)) * 0.01, q3_values, marker='o', color='r', label='q3 (mm)', markersize=1)
ax3.set_title('Joint Angle q3')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('q3 (mm)')
ax3.legend()


plt.tight_layout()
plt.show()

# บันทึกข้อมูลเป็น .csv
joint_angles_df = pd.DataFrame({
    'Time (s)': np.arange(len(q1_values)) * 0.01,
    'q1_values_deg': q1_values,
    'q2_values_deg': q2_values,
    'q3_values_mm': q3_values
})
joint_angles_df.to_csv('q.csv', index=False)
print("Saved interpolated joint angles to 'q.csv'")




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# กำหนดมวล (kg)
m_base = 13.521
m_link1 = 15.936
m_link2 = 13.373
m_shaft = 0.80683
# กำหนดความเฉื่อย (kg·m²)
I_base = np.array([
    [0.07366, 2.366E-19, -3.8224E-18],
    [2.366E-19, 0.08182, -0.0035749],
    [-3.8224E-18, -0.0035749, 0.056323]
])
I_link1 = np.array([
    [0.069528, -0.0021755, -0.059062],
    [-0.0021755, 0.26893, -0.00064495],
    [-0.059062, -0.00064495, 0.24013]
])
I_link2 = np.array([
    [0.060294, -6.133E-05, -0.049853],
    [-6.133E-05, 0.25254, -1.6152E-05],
    [-0.049853, -1.6152E-05, 0.21224]
])
I_shaft = np.array([
    [0.012728, 2.1641E-20, 6.244E-19],
    [2.1641E-20, 0.012728, 7.3305E-20],
    [6.244E-19, 7.3305E-20, 0.00015918]
])
# กำหนดค่า g (ความเร่งโน้มถ่วง)
g = 9.81  # m/s²
# อ่านข้อมูลจาก .csv
joint_angles_df = pd.read_csv('q_with_derivatives.csv')
# แปลงค่าจาก degrees เป็น radians สำหรับ q1 และ q2 ถ้าจำเป็น
joint_angles_df['q1_values_rad'] = np.deg2rad(joint_angles_df['q1_values_deg'])
joint_angles_df['q2_values_rad'] = np.deg2rad(joint_angles_df['q2_values_deg'])
# กำหนดตำแหน่งข้อต่อ (radians สำหรับ revolute, meters สำหรับ prismatic)
q1_values = joint_angles_df['q1_values_rad'].values       # radians
q2_values = joint_angles_df['q2_values_rad'].values       # radians
# ความเร็วของข้อต่อ
q1_dot_values = np.deg2rad(joint_angles_df['q1_dot_deg_per_s'].values)      # rad/s
q2_dot_values = np.deg2rad(joint_angles_df['q2_dot_deg_per_s'].values)      # rad/s
# การเร่งของข้อต่อ
q1_dotdot_values = np.deg2rad(joint_angles_df['q1_dotdot_deg_per_s2'].values)  # rad/s²
q2_dotdot_values = np.deg2rad(joint_angles_df['q2_dotdot_deg_per_s2'].values)  # rad/s²
# สร้างฟังก์ชันเพื่อคำนวณ Torque สำหรับชุดค่าเฉพาะ
def calculate_torque(q1, q2, q1_dot, q2_dot, q1_dotdot, q2_dotdot):
    # คำนวณ Mass Matrix
    a = I_link1[0, 0] + I_link2[0, 0] + I_shaft[0, 0] + m_link1 * (0.094352**2 + 0.0010294**2) + m_link2 * (0.19698**2 + 6.5485E-05**2)
    b = I_link2[0, 0] + I_shaft[0, 0] + m_link2 * (0.19698**2 + 6.5485E-05**2)
    M = np.array([
        [a + 2 * b * np.cos(q2), b * np.cos(q2)],
        [b * np.cos(q2), b]
    ])
    # คำนวณ Coriolis and Centrifugal Forces
    C = np.array([
        [-b * np.sin(q2) * q2_dot, -b * np.sin(q2) * (q1_dot + q2_dot)],
        [b * np.sin(q2) * q1_dot, 0],
    ])
    Cq_dot = C @ np.array([q1_dot, q2_dot])
    # คำนวณ Gravitational Forces
    G1 = m_base * g * 0.097334 + m_link1 * g * 0.12084 + m_link2 * g * 0.083139
    G2 = m_link1 * g * 0.12084 + m_link2 * g * 0.083139
    G = np.array([G1, G2])
    # Joint Accelerations
    q_ddot = np.array([q1_dotdot, q2_dotdot])
    # คำนวณ Torque
    tau = M @ q_ddot + Cq_dot + G
    return tau
# คำนวณ Torque สำหรับแต่ละชุดค่า
torques = []
for i in range(len(q1_values)):
    tau = calculate_torque(
        q1_values[i], q2_values[i],
        q1_dot_values[i], q2_dot_values[i],
        q1_dotdot_values[i], q2_dotdot_values[i]
    )
    torques.append(tau)

# แปลงเป็น numpy array
torques = np.array(torques)

# เพิ่มค่า Torque ลงใน DataFrame
joint_angles_df['tau1'] = torques[:, 0]
joint_angles_df['tau2'] = torques[:, 1]

# บันทึกข้อมูลเป็น .csv
joint_angles_df.to_csv('q_with_derivatives_and_torque.csv', index=False)
print("Saved joint angles, velocities, accelerations, and torques to 'q_with_derivatives_and_torque.csv'")

# พล็อต Torque
fig, (ax10, ax11) = plt.subplots(2, 1, figsize=(10, 8))

# Plot tau1
ax10.plot(joint_angles_df['Time (s)'], joint_angles_df['tau1'], color='c', label='tau1 (Nm)')
ax10.set_title('Torque of q1')
ax10.set_xlabel('Time (s)')
ax10.set_ylabel('tau1 (Nm)')
ax10.legend()

# Plot tau2
ax11.plot(joint_angles_df['Time (s)'], joint_angles_df['tau2'], color='m', label='tau2 (Nm)')
ax11.set_title('Torque of q2')
ax11.set_xlabel('Time (s)')
ax11.set_ylabel('tau2 (Nm)')
ax11.legend()

# # Plot tau3
# ax12.plot(joint_angles_df['Time (s)'], joint_angles_df['tau3'], color='y', label='tau3 (Nm)')
# ax12.set_title('Torque of q3')
# ax12.set_xlabel('Time (s)')
# ax12.set_ylabel('tau3 (Nm)')
# ax12.legend()

plt.tight_layout()
plt.show()












# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# # กำหนดมวล (kg)
# m_base = 13.521
# m_link1 = 15.936
# m_link2 = 13.373
# m_shaft = 0.80683
# # กำหนดความเฉื่อย (kg·m²)
# I_base = np.array([
#     [0.07366, 2.366E-19, -3.8224E-18],
#     [2.366E-19, 0.08182, -0.0035749],
#     [-3.8224E-18, -0.0035749, 0.056323]
# ])
# I_link1 = np.array([
#     [0.069528, -0.0021755, -0.059062],
#     [-0.0021755, 0.26893, -0.00064495],
#     [-0.059062, -0.00064495, 0.24013]
# ])
# I_link2 = np.array([
#     [0.060294, -6.133E-05, -0.049853],
#     [-6.133E-05, 0.25254, -1.6152E-05],
#     [-0.049853, -1.6152E-05, 0.21224]
# ])
# I_shaft = np.array([
#     [0.012728, 2.1641E-20, 6.244E-19],
#     [2.1641E-20, 0.012728, 7.3305E-20],
#     [6.244E-19, 7.3305E-20, 0.00015918]
# ])
# # กำหนดค่า g (ความเร่งโน้มถ่วง)
# g = 9.81  # m/s²
# # อ่านข้อมูลจาก .csv
# joint_angles_df = pd.read_csv('q_with_derivatives.csv')
# # แปลงค่าจาก degrees เป็น radians สำหรับ q1 และ q2 ถ้าจำเป็น
# joint_angles_df['q1_values_rad'] = np.deg2rad(joint_angles_df['q1_values_deg'])
# joint_angles_df['q2_values_rad'] = np.deg2rad(joint_angles_df['q2_values_deg'])
# # กำหนดตำแหน่งข้อต่อ (radians สำหรับ revolute, meters สำหรับ prismatic)
# q1_values = joint_angles_df['q1_values_rad'].values       # radians
# q2_values = joint_angles_df['q2_values_rad'].values       # radians
# q3_values = joint_angles_df['q3_values_mm'].values / 1000.0  # meters
# # ความเร็วของข้อต่อ
# q1_dot_values = np.deg2rad(joint_angles_df['q1_dot_deg_per_s'].values)      # rad/s
# q2_dot_values = np.deg2rad(joint_angles_df['q2_dot_deg_per_s'].values)      # rad/s
# q3_dot_values = joint_angles_df['q3_dot_mm_per_s'].values / 1000.0          # m/s
# # การเร่งของข้อต่อ
# q1_dotdot_values = np.deg2rad(joint_angles_df['q1_dotdot_deg_per_s2'].values)  # rad/s²
# q2_dotdot_values = np.deg2rad(joint_angles_df['q2_dotdot_deg_per_s2'].values)  # rad/s²
# q3_dotdot_values = joint_angles_df['q3_dotdot_mm_per_s2'].values / 1000.0      # m/s²
# # สร้างฟังก์ชันเพื่อคำนวณ Torque สำหรับชุดค่าเฉพาะ
# def calculate_torque(q1, q2, q3, q1_dot, q2_dot, q3_dot, q1_dotdot, q2_dotdot, q3_dotdot):
#     # คำนวณ Mass Matrix
#     a = I_link1[0, 0] + I_link2[0, 0] + I_shaft[0, 0] + m_link1 * (0.094352**2 + 0.0010294**2) + m_link2 * (0.19698**2 + 6.5485E-05**2)
#     b = I_link2[0, 0] + I_shaft[0, 0] + m_link2 * (0.19698**2 + 6.5485E-05**2)
#     c = I_shaft[0, 0] + m_shaft * (0.18237**2)
#     M = np.array([
#         [a + 2 * b * np.cos(q2), b * np.cos(q2), 0],
#         [b * np.cos(q2), b, 0],
#         [0, 0, c]
#     ])
#     # คำนวณ Coriolis and Centrifugal Forces
#     C = np.array([
#         [-b * np.sin(q2) * q2_dot, -b * np.sin(q2) * (q1_dot + q2_dot), 0],
#         [b * np.sin(q2) * q1_dot, 0, 0],
#         [0, 0, 0]
#     ])
#     Cq_dot = C @ np.array([q1_dot, q2_dot, q3_dot])
#     # คำนวณ Gravitational Forces
#     G1 = m_base * g * 0.097334 + m_link1 * g * 0.12084 + m_link2 * g * 0.083139
#     G2 = m_link1 * g * 0.12084 + m_link2 * g * 0.083139
#     G3 = m_shaft * g * 0.18237
#     G = np.array([G1, G2, G3])
#     # Joint Accelerations
#     q_ddot = np.array([q1_dotdot, q2_dotdot, q3_dotdot])
#     # คำนวณ Torque
#     tau = M @ q_ddot + Cq_dot + G
#     return tau
# # คำนวณ Torque สำหรับแต่ละชุดค่า
# torques = []
# for i in range(len(q1_values)):
#     tau = calculate_torque(
#         q1_values[i], q2_values[i], q3_values[i],
#         q1_dot_values[i], q2_dot_values[i], q3_dot_values[i],
#         q1_dotdot_values[i], q2_dotdot_values[i], q3_dotdot_values[i]
#     )
#     torques.append(tau)

# # แปลงเป็น numpy array
# torques = np.array(torques)

# # เพิ่มค่า Torque ลงใน DataFrame
# joint_angles_df['tau1'] = torques[:, 0]
# joint_angles_df['tau2'] = torques[:, 1]
# joint_angles_df['tau3'] = torques[:, 2]

# # บันทึกข้อมูลเป็น .csv
# joint_angles_df.to_csv('q_with_derivatives_and_torque.csv', index=False)
# print("Saved joint angles, velocities, accelerations, and torques to 'q_with_derivatives_and_torque.csv'")

# # พล็อต Torque
# fig, (ax10, ax11, ax12) = plt.subplots(3, 1, figsize=(10, 8))

# # Plot tau1
# ax10.plot(joint_angles_df['Time (s)'], joint_angles_df['tau1'], color='c', label='tau1 (Nm)')
# ax10.set_title('Torque of q1')
# ax10.set_xlabel('Time (s)')
# ax10.set_ylabel('tau1 (Nm)')
# ax10.legend()

# # Plot tau2
# ax11.plot(joint_angles_df['Time (s)'], joint_angles_df['tau2'], color='m', label='tau2 (Nm)')
# ax11.set_title('Torque of q2')
# ax11.set_xlabel('Time (s)')
# ax11.set_ylabel('tau2 (Nm)')
# ax11.legend()

# # Plot tau3
# ax12.plot(joint_angles_df['Time (s)'], joint_angles_df['tau3'], color='y', label='tau3 (Nm)')
# ax12.set_title('Torque of q3')
# ax12.set_xlabel('Time (s)')
# ax12.set_ylabel('tau3 (Nm)')
# ax12.legend()

# plt.tight_layout()
# plt.show()






