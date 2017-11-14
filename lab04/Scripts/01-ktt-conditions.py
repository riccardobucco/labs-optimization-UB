import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

INTERVAL_START_X1 = -10
INTERVAL_STOP_X1 = 10
INTERVAL_START_X2 = -10
INTERVAL_STOP_X2 = 10
SAMPLING_INTERVAL = 0.1

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
X1_line = np.linspace(INTERVAL_START_X1, INTERVAL_STOP_X1, ((INTERVAL_STOP_X1 - INTERVAL_START_X1) / SAMPLING_INTERVAL) + 1)
X2_line = 3 - X1_line
X1 = X1_line
X2 = np.linspace(min(X2_line), max(X2_line), ((max(X2_line) - min(X2_line)) / SAMPLING_INTERVAL) + 1)
X1, X2 = np.meshgrid(X1, X2)
Y = 0.5*X1**2 + 0.5*X2**2 - X1*X2 - 3*X2
levels_number = 50
ax.contourf(X1, X2, Y, levels_number, cmap='inferno')
# Plot a line representing the constraint
ax.plot(X1_line, X2_line, 'r')
# Plot the point that minimized the constrainted function
ax.plot(0.75, 2.25, 'ro')
# Save the current figure
plt.savefig('../Images/01-function-with-constraints.png')