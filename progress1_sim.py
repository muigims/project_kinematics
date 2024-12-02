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



#-------------------------------------------------------------------------------------------------
# import roboticstoolbox as rtb
# import numpy as np
# from math import pi

# # แปลงค่าจากมิลลิเมตรเป็นเมตร
# a_1 = 0.145      # ความยาวของแขนแรก (100 มม. / 1000 = 0.1 เมตร)
# a_2 = 0.145      # ความยาวของแขนที่สอง (100 มม. / 1000 = 0.1 เมตร)
# d3_min = 0.0   # ตำแหน่ง Z ต่ำสุดของข้อต่อ Prismatic (เมตร)
# d3_max = 0.1   # ตำแหน่ง Z สูงสุดของข้อต่อ Prismatic (100 มม. / 1000 = 0.1 เมตร)

# # สร้างหุ่นยนต์ SCARA ด้วยหน่วยเมตร
# robot = rtb.DHRobot([
#     # ข้อต่อที่หนึ่ง: Revolute Joint (หมุนรอบแกน Z)
#     rtb.RevoluteMDH(alpha=0.0, a=a_1, d=0.0, offset=0.0, qlim=[-pi, pi]),
#     # ข้อต่อที่สอง: Revolute Joint (หมุนรอบแกน Z)
#     rtb.RevoluteMDH(alpha=0.0, a=a_2, d=0.0, offset=0.0, qlim=[-pi, pi]),
#     # ข้อต่อที่สาม: Prismatic Joint (เคลื่อนที่ตามแกน Z)
#     rtb.PrismaticMDH(alpha=0.0, a=0.0, theta=0.0, offset=0.0, qlim=[d3_min, d3_max]),
# ], name='SCARA Robot')

# from spatialmath import SE3

# # ตำแหน่งเป้าหมาย (แปลงเป็นเมตร)
# x_target = 0.1  # 110 มม. / 1000 = 0.110 เมตร
# y_target = 0.1  # 110 มม. / 1000 = 0.110 เมตร
# z_target = 0.1  # 1 มม. / 1000 = 0.001 เมตร

# # ฟังก์ชัน inverse kinematics แบบเชิงวิเคราะห์สำหรับ SCARA
# def scara_ik(x, y, z, a1, a2):
#     r = np.hypot(x, y)
#     cos_q2 = (r**2 - a1**2 - a2**2) / (2 * a1 * a2)
#     # ตรวจสอบว่าค่า cos_q2 อยู่ในช่วง [-1, 1] หรือไม่
#     if abs(cos_q2) > 1.0:
#         print("ตำแหน่งเป้าหมายอยู่นอกขอบเขตการเข้าถึง")
#         return None
#     sin_q2 = np.sqrt(1 - cos_q2**2)
#     q2_options = [np.arctan2(sin_q2, cos_q2), np.arctan2(-sin_q2, cos_q2)]
#     solutions = []
#     for q2 in q2_options:
#         k1 = a1 + a2 * np.cos(q2)
#         k2 = a2 * np.sin(q2)
#         q1 = np.arctan2(y, x) - np.arctan2(k2, k1)
#         solutions.append((q1, q2))
#     return solutions

# # เรียกใช้ฟังก์ชัน inverse kinematics
# solutions = scara_ik(x_target, y_target, z_target, a_1, a_2)

# if solutions is None:
#     print("ตำแหน่งเป้าหมายไม่สามารถเข้าถึงได้")
# else:
#     found_solution = False
#     for idx, (q1, q2) in enumerate(solutions):
#         d3 = z_target  # การเคลื่อนที่ตามแกน Z
#         q = np.array([q1, q2, d3])
#         # ตรวจสอบว่ามุมข้อต่ออยู่ในขอบเขตหรือไม่
#         if robot.islimit(q):
#             print(f"คำตอบที่ {idx+1} อยู่นอกขอบเขตของข้อต่อ")
#         else:
#             found_solution = True
#             print(f"คำตอบที่ {idx+1}:")
#             print(f"q1 = {np.degrees(q1):.2f} องศา")
#             print(f"q2 = {np.degrees(q2):.2f} องศา")
#             print(f"d3 = {d3:.4f} เมตร")
#             # ตรวจสอบตำแหน่งด้วย forward kinematics
#             T = robot.fkine(q)
#             print(f"ตำแหน่งที่ได้จาก forward kinematics: {T.t}")
#     if not found_solution:
#         print("ไม่มีคำตอบที่อยู่ในขอบเขตของข้อต่อ")

# import numpy as np
# import matplotlib.pyplot as plt

# # ความยาวแขนหุ่นยนต์
# a1 = 0.145  # เมตร
# a2 = 0.145  # เมตร

# # สร้างกริดของมุมข้อต่อ
# theta1 = np.linspace(np.pi, 0, 300)
# theta2 = np.linspace(30, 0, 300)
# Theta1, Theta2 = np.meshgrid(theta1, theta2)

# # คำนวณตำแหน่งปลายแขนหุ่นยนต์ในระนาบ XY
# X = a1 * np.cos(Theta1) + a2 * np.cos(Theta1 + Theta2)
# Y = a1 * np.sin(Theta1) + a2 * np.sin(Theta1 + Theta2)

# # แสดงพื้นที่การทำงานในระนาบ XY
# plt.figure(figsize=(8,8))
# plt.plot(X, Y, '.', markersize=1)
# plt.xlabel('X (เมตร)')
# plt.ylabel('Y (เมตร)')
# plt.title('พื้นที่การทำงานของหุ่นยนต์ SCARA ในระนาบ XY')
# plt.axis('equal')
# plt.grid(True)
# plt.show()


import roboticstoolbox as rtb
import numpy as np
import matplotlib.pyplot as plt

# ---- Robot Parameters ----
# ความยาวของแขนหุ่นยนต์ (เมตร)
# a_1 = 0.145      # ความยาวของแขนแรก (100 มม.)
# a_2 = 0.145      # ความยาวของแขนที่สอง (100 มม.)
d3_min = 0.0   # ตำแหน่ง Z ต่ำสุดของข้อต่อ Prismatic (0 มม.)
d3_max = 0.2   # ตำแหน่ง Z สูงสุดของข้อต่อ Prismatic (100 มม.)

a_1 = 0.320      # ความยาวของแขนแรก (100 มม.)
a_2 = 0.320      # ความยาวของแขนที่สอง (100 มม.)

# สร้างหุ่นยนต์ SCARA ด้วย MDH parameters
robot = rtb.DHRobot([
    rtb.RevoluteMDH(alpha=0.0, a=a_1, d=0.0, offset=0.0, qlim=[-np.pi, np.pi]),
    rtb.RevoluteMDH(alpha=0.0, a=a_2, d=0.0, offset=0.0, qlim=[-np.pi, np.pi]),
    rtb.PrismaticMDH(alpha=0.0, a=0.0, theta=0.0, offset=0.0, qlim=[d3_min, d3_max]),
], name='SCARA Robot')

# ---- Generate Workspace ----
# สร้างกริดของมุมข้อต่อ
theta1 = np.linspace(np.pi, 0, 300)
theta2 = np.linspace(30, 0, 300)
Theta1, Theta2 = np.meshgrid(theta1, theta2)

# คำนวณตำแหน่งปลายแขนหุ่นยนต์ในระนาบ XY
X = a_1 * np.cos(Theta1) + a_2 * np.cos(Theta1 + Theta2)
Y = a_1 * np.sin(Theta1) + a_2 * np.sin(Theta1 + Theta2)

# ---- Use Coordinates from optimal_path ----
# สมมติว่า optimal_path มีอยู่แล้ว และเป็นลิสต์ของจุด [(x1, y1, z1), (x2, y2, z2), ...]

# แยกพิกัด x, y, z
x_coords = [point[0] for point in optimal_path]
y_coords = [point[1] for point in optimal_path]
z_coords = [point[2] for point in optimal_path]

x_coords = np.array(x_coords) / 1000.0  # แปลงเป็นเมตร
y_coords = np.array(y_coords) / 1000.0
z_coords = np.array(z_coords) / 1000.0
# หากพิกัดเป็นมิลลิเมตรและต้องการแปลงเป็นเมตร ให้ใช้:
# x_coords = [x / 1000.0 for x in x_coords]
# y_coords = [y / 1000.0 for y in y_coords]
# z_coords = [z / 1000.0 for z in z_coords]

# ---- Visualization ----
plt.figure(figsize=(8, 8))
plt.plot(X, Y, '.', markersize=1, label='พื้นที่การทำงาน')  # พื้นที่การทำงาน
plt.plot(x_coords, y_coords, c='r', label='เส้นทางการเคลื่อนที่')  # เส้นทางจาก optimal_path
plt.xlabel('X (เมตร)')
plt.ylabel('Y (เมตร)')
plt.title('พื้นที่การทำงานของหุ่นยนต์ SCARA พร้อมเส้นทางการเคลื่อนที่')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

# ฟังก์ชันสำหรับคำนวณ Inverse Kinematics
def calculate_ik(x_coords, y_coords, z_coords, a_1, a_2, z_base):
    joint_trajectory = []  # สำหรับเก็บค่ามุมข้อต่อ (q1, q2, d3)
    for x, y, z in zip(x_coords, y_coords, z_coords):
        r = np.hypot(x, y)  # คำนวณระยะ r = sqrt(x^2 + y^2)
        
        # คำนวณ cos และ sin ของ q2
        cos_q2 = (r**2 - a_1**2 - a_2**2) / (2 * a_1 * a_2)
        if abs(cos_q2) > 1.0:
            print(f"ตำแหน่งเป้าหมาย ({x:.2f}, {y:.2f}, {z:.2f}) อยู่นอกขอบเขตการเข้าถึง")
            joint_trajectory.append(None)
            continue
        sin_q2 = np.sqrt(1 - cos_q2**2)

        # มีตัวเลือกสำหรับ q2 (สองค่า)
        q2_options = [np.arctan2(sin_q2, cos_q2), np.arctan2(-sin_q2, cos_q2)]

        # เลือก q2 ที่เหมาะสม
        for q2 in q2_options:
            k1 = a1 + a2 * np.cos(q2)
            k2 = a2 * np.sin(q2)
            q1 = np.arctan2(y, x) - np.arctan2(k2, k1)  # คำนวณ q1
            d3 = z - z_base  # คำนวณ d3

            # เก็บค่ามุมข้อต่อ (q1, q2, d3)
            joint_trajectory.append((q1, q2, d3))
            break  # ใช้ตัวเลือก q2 แรกที่เหมาะสม
    return joint_trajectory

# เรียกใช้ฟังก์ชัน IK
joint_trajectory = calculate_ik(x_coords, y_coords, z_coords, a_1, a_2, z_base)

# แสดงผลลัพธ์
for i, joint in enumerate(joint_trajectory):
    if joint is not None:
        q1, q2, d3 = joint
        print(f"จุดที่ {i+1}:")
        print(f"  q1 = {np.degrees(q1):.2f} องศา")
        print(f"  q2 = {np.degrees(q2):.2f} องศา")
        print(f"  d3 = {d3:.3f} เมตร")
    else:
        print(f"จุดที่ {i+1}: ไม่สามารถเข้าถึงได้")