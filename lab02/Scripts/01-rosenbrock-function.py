import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

INTERVAL_START_X1 = -2
INTERVAL_STOP_X1 = 2
INTERVAL_START_X2 = -1
INTERVAL_STOP_X2 = 3
SAMPLING_INTERVAL = 0.05
A = 1
B = 100
MAX_ITERATIONS = 1000
THRESHOLDS = np.array([0.001, 0.00001, 0.0000001])
THRESHOLDS_NAMES = np.array(['001', '00001', '0000001'])
STARTING_POINTS = np.array([[-1.5,-0.5], [0.5, 2.5]])
COLORS = np.array(['g', 'y'])

# Function that computes the value of the function in a point x=(x1,x2)
def myFunction(x):
    y = (A - x[0])**2 + B*(x[1] - x[0]**2)**2
    return y

# Function that computes the gradient (gx1, gx2) of the function in a point x=(x1,x2)
def myFunctionGradient(x):
    gradient = np.array([2*x[0] - 2*A + B*4*x[0]**3 - 4*B*x[1]*x[0], 2*B*x[1] - 2*B*x[0]**2])
    return gradient

# Function that finds an alpha such that f(xk - alpha*grad(f(xk))) < f(xk)
# alpha is found starting from 1 and dividing iteratively by 2 until the condition is satisfied 
def findAlpha(xk, function, function_gradient):
    alpha = 1.0
    while function(xk - alpha*function_gradient(xk)) >= function(xk):
        alpha = alpha / 2
    return alpha

# Function that implements the gradient descent method, with alpha that changes at every iteration.
# It requires a starting point, alpha, a function that computes the value of a function in a specified point,
# a function that computes the gradient of a function in a specified point, the maximum number of iterations
# to perform and a threshold to stop the method.
# It returns an array of points, where each point is computed using the gradient descent method.
def gradientDescentWithDynamicAlpha(x0, function, function_gradient, max_iterations, threshold):
    points = [x0]
    xk = x0
    alpha = findAlpha(xk, function, function_gradient)
    xk1 = xk - alpha*function_gradient(xk)
    i = 1
    while (i < max_iterations) & (abs(function(xk1) - function(xk)) > threshold):
        xk = xk1
        alpha = findAlpha(xk, function, function_gradient)
        xk1 = xk - alpha*function_gradient(xk)
        points.append(xk1)
        i = i + 1
    return np.array(points)

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
Y = (A - X1)**2 + B*(X2 - X1**2)**2
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='inferno')
# Save the current figure
plt.savefig('../Images/01-rosenbrock-function-surface.png')

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
# Plot the global minimum of the function
ax.plot(A, A**2, 'ro')
# Save the current figure
plt.savefig('../Images/01-rosenbrock-function-contours.png')

# Apply the gradient method to the function starting at the specified points
for threshold_index, threshold in enumerate(THRESHOLDS):
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
    # Plot the global minimum of the function
    ax.plot(A, A**2, 'ro')
    for i in range(0, COLORS.size):
        x0 = STARTING_POINTS[i]
        points = gradientDescentWithDynamicAlpha(x0, myFunction, myFunctionGradient, MAX_ITERATIONS, threshold)
        print(points.shape)
        ax.plot(points[:,0], points[:,1], COLORS[i])
    # Save the current figure
    plt.savefig('../Images/01-rosenbrock-function-contours-threshold-'+THRESHOLDS_NAMES[threshold_index]+'.png')