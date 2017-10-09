import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

INTERVAL_START = -5
INTERVAL_STOP = 5
SAMPLING_INTERVAL = 0.1

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
X1 = np.linspace(INTERVAL_START, INTERVAL_STOP, ((INTERVAL_STOP - INTERVAL_START) / SAMPLING_INTERVAL) + 1)
X2 = np.linspace(INTERVAL_START, INTERVAL_STOP, ((INTERVAL_STOP - INTERVAL_START) / SAMPLING_INTERVAL) + 1)
X1, X2 = np.meshgrid(X1, X2)
Y = X1**2 + X2**2
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='inferno')
# Save the current figure
plt.savefig('../Images/02-two-dimensional-function-1-surface.png')

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
plt.savefig('../Images/02-two-dimensional-function-1-contours.png')

# Compute the derivatives of the 2D function with respect to each dimension
gradient_X1, gradient_X2 = np.gradient(Y)

# Create a new figure
fig = plt.figure()
# Split the figure in 1*1 (nrows*ncols) subaxes and create a new suplot positioned at 1 (plot_number)
nrows = 1
ncols = 1
plot_number = 1
ax_gradient_X1 = fig.add_subplot(nrows, ncols, plot_number, projection='3d')
# Set the labels for the axes
ax_gradient_X1.set_xlabel('$x_1$')
ax_gradient_X1.set_ylabel('$x_2$')
ax_gradient_X1.set_zlabel(r'$\frac{\partial f}{\partial x_1}(x_1, x_2)$')
# Rotate the axes (azim is the azimuth angle in the x,y plane)
ax_gradient_X1.view_init(azim=330)
# Plot a surface with a solid linestyle connecting all the vertices
ax_gradient_X1.plot_surface(X1, X2, gradient_X1, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='inferno')
# Save the current figure
plt.savefig('../Images/02-two-dimensional-function-1-gradient-X1.png')

# Create a new figure
fig = plt.figure()
# Split the figure in 1*1 (nrows*ncols) subaxes and create a new suplot positioned at 1 (plot_number)
nrows = 1
ncols = 1
plot_number = 1
ax_gradient_X2 = fig.add_subplot(nrows, ncols, plot_number, projection='3d')
# Set the labels for the axes
ax_gradient_X2.set_xlabel('$x_1$')
ax_gradient_X2.set_ylabel('$x_2$')
ax_gradient_X2.set_zlabel(r'$\frac{\partial f}{\partial x_1}(x_1, x_2)$')
# Rotate the axes (azim is the azimuth angle in the x,y plane)
ax_gradient_X2.view_init(azim=330)
# Plot a surface with a solid linestyle connecting all the vertices
ax_gradient_X2.plot_surface(X1, X2, gradient_X2, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='inferno')
plt.savefig('../Images/02-two-dimensional-function-1-gradient-X2.png')

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
ax.set_zlabel('$f_A(x_1, x_2)$')
# Plot a surface with a solid linestyle connecting all the vertices
Y = - X1**2 - X2**2
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='inferno')
plt.savefig('../Images/02-two-dimensional-function-A-surface.png')

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
plt.savefig('../Images/02-two-dimensional-function-A-contours.png')

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
ax.set_zlabel('$f_B(x_1, x_2)$')
# Plot a surface with a solid linestyle connecting all the vertices
Y = X1**2 - X2**2
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='inferno')
plt.savefig('../Images/02-two-dimensional-function-B-surface.png')

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
plt.savefig('../Images/02-two-dimensional-function-B-contours.png')

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
ax.set_zlabel('$f_C(x_1, x_2)$')
# Plot a surface with a solid linestyle connecting all the vertices
Y = X1**2
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='inferno')
plt.savefig('../Images/02-two-dimensional-function-C-surface.png')

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
plt.savefig('../Images/02-two-dimensional-function-C-contours.png')