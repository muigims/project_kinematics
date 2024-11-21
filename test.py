import numpy as np

optimal_path = np.array(optimal_path) / 1000
trajectory_points = np.array(trajectory_points) / 1000

print(optimal_path)
print(trajectory_points)

# Define the robot using DH parameters
robot = rtb.DHRobot([
    rtb.RevoluteMDH(d=0.103), 
    rtb.RevoluteMDH(a=0, alpha=pi/2),  
    rtb.RevoluteMDH(a=0.135),   
    rtb.RevoluteMDH(a=0.147) 
], name="3R_robot")


# Define joint angles
q = [pi/2, pi/2, pi, pi/2]  # e.g., all joints at 45 degrees

# Compute forward kinematics
T = robot.fkine(q)

print(T)
fig = robot.plot(q, block=False)


def custom_ikine(robot, T_desired, initial_guess):
    # Define the objective function
    def objective(q):
        T_actual = robot.fkine(q)
        return np.linalg.norm(T_actual.A - T_desired.A)

    # Run the optimization
    result = minimize(objective, initial_guess, bounds=[(-pi, pi) for _ in initial_guess])
    
    return result.x