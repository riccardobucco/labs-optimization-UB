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
plt.savefig('../Images/01-two-dimensional-function.png')