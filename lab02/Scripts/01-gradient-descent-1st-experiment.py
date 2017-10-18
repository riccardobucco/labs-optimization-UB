import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

INTERVAL_START_X1 = -5
INTERVAL_STOP_X1 = 5
INTERVAL_START_X2 = -5
INTERVAL_STOP_X2 = 5
SAMPLING_INTERVAL = 0.01
ITERATIONS = 100
ALPHA = 0.1
STARTING_POINTS = np.array([[4,4], [-3,2], [4,-3]])
COLORS = np.array(['g', 'r', 'y'])

# Function that computes the value of the function in a point x=(x1,x2)
def myFunction(x):
    y = x[0]**2 + x[1]**2
    return y

# Function that computes the gradient (gx1, gx2) of the function in a point x=(x1,x2)
def myFunctionGradient(x):
    gradient = np.array([2*x[0], 2*x[1]])
    return gradient

# Function that implements the gradient descent method.
# It requires a starting point, alpha, a function that computes the value of a function in a specified point,
# a function that computes the gradient of a function in a specified point, the number of iterations to perform
# It returns an array of points, where each point is computed using the gradient descent method.
def gradientDescent(x0, alpha, function, function_gradient, iterations):
    points = [x0]
    xk = x0
    gradient_xk = function_gradient(xk)
    for i in range(0, iterations):
        xk = xk - alpha*gradient_xk
        gradient_xk = function_gradient(xk)
        points.append(xk)
    return np.array(points)

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
X1 = np.linspace(INTERVAL_START_X1, INTERVAL_STOP_X1, ((INTERVAL_STOP_X1 - INTERVAL_START_X1) / SAMPLING_INTERVAL) + 1)
X2 = np.linspace(INTERVAL_START_X2, INTERVAL_STOP_X2, ((INTERVAL_STOP_X2 - INTERVAL_START_X2) / SAMPLING_INTERVAL) + 1)
X1, X2 = np.meshgrid(X1, X2)
Y = X1**2 + X2**2
levels_number = 100
ax.contourf(X1, X2, Y, levels_number, cmap='inferno')

# Apply the gradient method to the function starting at the specified points
for i in range(0, COLORS.size):
    x0 = STARTING_POINTS[i]
    points = gradientDescent(x0, ALPHA, myFunction, myFunctionGradient, ITERATIONS)
    ax.plot(points[:,0], points[:,1], COLORS[i]+'o')

# Save the current figure
plt.savefig('../Images/01-gradient-descent-1st-experiment.png')