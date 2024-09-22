import matplotlib.pyplot as plt
import numpy as np

# Load points and colors from the first set of CSV files
points1 = np.loadtxt('respoints15015.csv', delimiter=',')
colors1 = np.loadtxt('rescolor15015.csv', delimiter=',')

# Load points and colors from the second set of CSV files
points2 = np.loadtxt('respoints10010.csv', delimiter=',')
colors2 = np.loadtxt('rescolor10010.csv', delimiter=',')

# Print shapes to ensure correct loading
print("Points1 shape:", points1.shape)
print("Colors1 shape:", colors1.shape)
print("Points2 shape:", points2.shape)
print("Colors2 shape:", colors2.shape)

# Reshape the data to match the structure used in the original code
num_rays = 1024  # Change this based on your actual data
num_points_per_ray = 60  # Change this based on your actual data

# Reshape and slice the first set of data
colors1 = colors1.reshape((num_rays, num_points_per_ray, 3))
rays_to_keep = np.any(colors1 > 1e-3, axis=1)

points1 = points1.reshape((num_rays, num_points_per_ray, 3))

# Reshape and slice the second set of data
points2 = points2.reshape((num_rays, num_points_per_ray, 3))
colors2 = colors2.reshape((num_rays, num_points_per_ray, 3))
points1 = points1[rays_to_keep]
colors1 = colors1[rays_to_keep]

points2 = points2[rays_to_keep]
colors2 = colors2[rays_to_keep]

# Set up the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each point in its respective color for the first dataset
for i, ray in enumerate(points1):
    ray_colors = colors1[i]
    ax.scatter(ray[:, 0], ray[:, 1], ray[:, 2], color=ray_colors, s=10, alpha=0.5)  # s=10 for point size

# Plot each point in its respective color for the second dataset
for i, ray in enumerate(points2):
    ray_colors = colors2[i]
    ax.scatter(ray[:, 0], ray[:, 1], ray[:, 2], color=ray_colors, s=10, alpha=0.5)  # s=10 for point size

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Combined Points on Rays with Specific Colors')

# Save the plot
plt.savefig('combined_plot1.png')
plt.show()
