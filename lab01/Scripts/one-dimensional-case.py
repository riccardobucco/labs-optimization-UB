import numpy as np
import matplotlib.pyplot as plt

INTERVAL_START = -2
INTERVAL_STOP = 2
SAMPLING_INTERVAL = 0.1

# Create a new figure
fig = plt.figure()
# Split the figure in 1*1 (nrows*ncols) subaxes and create a new suplot positioned at 1 (plot_number)
nrows = 1
ncols = 1
plot_number = 1
ax = fig.add_subplot(nrows, ncols, plot_number)
# Set the labels for the axes
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
# Plot a line with a solid linestyle connecting all the vertices
X = np.linspace(INTERVAL_START, INTERVAL_STOP, ((INTERVAL_STOP - INTERVAL_START) / SAMPLING_INTERVAL) + 1)
Y = X**3 - 2*X + 2
ax.plot(X, Y)
# Save the current figure
plt.savefig('../Images/01-one-dimensional-function-example.png')

# Add two dashed lines indicating where the (analitically) computed extrema are located
X = [0.8165, 0.8165]
Y = [Y[0], Y[-1]] # Y[-1] is the last element of the Y array
ax.plot(X, Y, color='red', linestyle='dashed')
X = [-0.8165,-0.8165]
ax.plot(X, Y, color='red', linestyle='dashed')
# Save the current figure
plt.savefig('../Images/01-one-dimensional-function-example-with-extrema')