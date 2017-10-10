import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

INTERVAL_START_X1 = -2
INTERVAL_STOP_X1 = 2
INTERVAL_START_X2 = -1
INTERVAL_STOP_X2 = 1
SAMPLING_INTERVAL = 0.01

# Create a new figure
fig = plt.figure()
# Split the figure in 1*1 (nrows*ncols) subaxes and create a new suplot positioned at 1 (plot_number)
nrows = 1
ncols = 1
plot_number = 1
ax = fig.add_subplot(nrows, ncols, plot_number, projection='3d')
# Set the labels for the axes
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1, x_2)$')
# Plot a surface with a solid linestyle connecting all the vertices
X1 = np.linspace(INTERVAL_START_X1, INTERVAL_STOP_X1, ((INTERVAL_STOP_X1 - INTERVAL_START_X1) / SAMPLING_INTERVAL) + 1)
X2 = np.linspace(INTERVAL_START_X2, INTERVAL_STOP_X2, ((INTERVAL_STOP_X2 - INTERVAL_START_X2) / SAMPLING_INTERVAL) + 1)
X1, X2 = np.meshgrid(X1, X2)
Y = X1**2 * (4 - 2.1 * X1**2 + 1/3 * X1**4) + X1*X2 + X2**2 * (-4 + 4 * X2**2)
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='inferno')
# Save the current figure
plt.savefig('../Images/03-exercise-function-surface.png')

# Create a new figure
fig = plt.figure()
# Split the figure in 1*1 (nrows*ncols) subaxes and create a new suplot positioned at 1 (plot_number)
nrows = 1
ncols = 1
plot_number = 1
ax = fig.add_subplot(nrows, ncols, plot_number)
# Set the labels for the axes
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
# Plot filled contours (up to levels_number automatically-chosen levels)
levels_number = 100
ax.contourf(X1, X2, Y, levels_number, cmap='inferno')
# Save the current figure
plt.savefig('../Images/03-exercise-function-contours.png')

# Compute the derivatives of the 2D function with respect to each dimension
gradient_X1, gradient_X2 = np.gradient(Y)

# Compute the squared norm of each point f'(x1, x2)
gradient_norm = np.zeros((gradient_X1.shape))
for i in range(0, gradient_X1.shape[0]):
    for j in range(0, gradient_X1.shape[1]):
        gradient_norm[i][j] = (np.linalg.norm([gradient_X1[i][j], gradient_X2[i][j]]))**2

# Using a brute force approach, find all the stationary points, compute their Hessian matrix and find out the respective eigenvalues
for i in range(1, gradient_X1.shape[0]-1):
    for j in range(1, gradient_X1.shape[1]-1):
        if (gradient_norm[i][j] == np.min(gradient_norm[i-1:i+2, j-1:j+2])):
            ax.plot(X1[i][j], X2[i][j], 'bo')
            hessian_matrix = [[10*X1[i][j]**4 - 25.2*X1[i][j]**2 + 8, 1], [1, 48*X2[i][j]**2 - 8]]
            eigenvalues, eigenvectors = np.linalg.eig(hessian_matrix)
            
# Save the current figure
plt.savefig('../Images/03-exercise-function-contours-with-stationary-points.png')