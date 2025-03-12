import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to read points from a CSV file
def read_points_csv(file_path):
    df = pd.read_csv(file_path)
    return df[['x', 'y', 'z']].values

# Function to apply rotation and translation to points
def transform_points(points, rotation_matrix, translation_vector):
    # Convert points to homogeneous coordinates (add a 1 as the fourth coordinate)
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # Apply rotation and translation
    transformed_points = np.dot(points, rotation_matrix.T) + translation_vector
    
    return transformed_points

# Define the full 4x4 transformation matrix provided
full_transformation_matrix = np.array([[ 0.4430,  0.3138, -0.8398, -3.3855],
         [-0.8965,  0.1550, -0.4149, -1.6727],
         [ 0.0000,  0.9368,  0.3500,  1.4108],
         [ 0.0000,  0.0000,  0.0000,  1.0000]])

# Extract rotation matrix (upper-left 3x3 submatrix)
rotation_matrix = full_transformation_matrix[:3, :3]

# Extract translation vector (last column of the 3x4 matrix)
translation_vector = full_transformation_matrix[:3, 3]

# File path to the CSV file
file_path = 'points0.csv'

# Read points from CSV file
points = read_points_csv(file_path)

# Apply the rotation and translation
transformed_points = transform_points(points, rotation_matrix, translation_vector)

# Separate the points into x, y, and z coordinates
x_original, y_original, z_original = points[:, 0], points[:, 1], points[:, 2]
x_transformed, y_transformed, z_transformed = transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2]
x_center, y_center, z_center = translation_vector

# Create a 3D plot for original, transformed points, and the center
fig = plt.figure()

# Plot original points
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x_original, y_original, z_original, c='b', marker='o', label='Original Points')
ax1.set_title('Original Points')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Plot transformed points and connect each to the center
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(x_transformed, y_transformed, z_transformed, c='r', marker='o', label='Transformed Points')
ax2.scatter(x_center, y_center, z_center, c='g', marker='x', s=100, label='Center Point')

# Draw lines from each transformed point to the center
for pt in transformed_points:
    ax2.plot([pt[0], x_center], [pt[1], y_center], [pt[2], z_center], 'k--', linewidth=1)

ax2.set_title('Transformed Points with Center Connections')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend()

# Show the plot
plt.savefig('points.png')
