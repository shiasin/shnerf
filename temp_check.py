import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to read the CSV files
def read_csv_points(file_path):
    points = []
    directions = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            point = [float(row[0]), float(row[1]), float(row[2])]
            points.append(point)
    return np.array(points)

# File paths for the two CSV files
file1 = 'pointsx.csv'  # Replace with actual file path
file2 = 'points.csv'  # Replace with actual file path

# Read points from both files
points1 = read_csv_points(file1)
points2 = read_csv_points(file2)

# Plot the points in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points from file1
ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], color='r', label='File1')

# Plot points from file2
ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], color='b', label='File2')

# Label the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set legend
ax.legend()

# Show the plot
plt.savefig('points1.png')
