import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)  
num_steps = 50  
dt = 0.1  

# True position in 3D space
true_x = np.linspace(0, 100, num_steps)  
true_y = np.linspace(0, 50, num_steps)   
true_z = np.linspace(0, 20, num_steps)   

# Simulated radar measurements with noise
measurement_noise = np.random.normal(0, 2, (num_steps, 3))  
measurements = np.vstack((true_x, true_y, true_z)).T + measurement_noise

# Kalman Filter Initialization
x_kalman = np.zeros((num_steps, 3))  
P = np.eye(3) * 10  

# Kalman Filter Matrix  
F = np.eye(3) 
H = np.eye(3)
Q = np.eye(3) * 0.1  
R = np.eye(3) * 4  

# Kalman Filter Loop
for k in range(num_steps):
    x_kalman[k] = F @ x_kalman[k-1] if k > 0 else measurements[k]
    P = F @ P @ F.T + Q
    y = measurements[k] - H @ x_kalman[k]  
    S = H @ P @ H.T + R  
    K = P @ H.T @ np.linalg.inv(S)  
    x_kalman[k] += K @ y
    P = (np.eye(3) - K @ H) @ P 

# 3D Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(true_x, true_y, true_z, 'g-', label="True Path")  
ax.scatter(measurements[:, 0], measurements[:, 1], measurements[:, 2], color='r', label="Noisy Measurements")  
ax.plot(x_kalman[:, 0], x_kalman[:, 1], x_kalman[:, 2], 'b-', label="Kalman Filter Estimate")  

ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("3D Target Tracking with Kalman Filter")
ax.legend()
plt.show()
