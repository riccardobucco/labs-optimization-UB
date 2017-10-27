import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

INTERVAL_START_X1 = -2
INTERVAL_STOP_X1 = 2
INTERVAL_START_X2 = -1.5
INTERVAL_STOP_X2 = 1.5
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
Y = (X1**2) * (4 - 2.1 * (X1**2) + 1.0/3 * (X1**4)) + X1*X2 + (X2**2) * (-4 + 4 * (X2**2))
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='inferno')
# Save the current figure
plt.savefig('../Images/01-gradient-descent-1st-lab-function-surface.png')

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
plt.savefig('../Images/01-gradient-descent-1st-lab-function-contours.png')