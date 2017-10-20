import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

INTERVAL_START_X1 = -2
INTERVAL_STOP_X1 = 2
INTERVAL_START_X2 = -1
INTERVAL_STOP_X2 = 1
SAMPLING_INTERVAL = 0.01
MAX_ITERATIONS = np.array([7, 4])
ALPHA = 0.05
THRESHOLD = 0.001
STARTING_POINTS = np.array([[1, 0], [0.6, -0.3]])
COLORS = np.array(['g', 'r'])

# Function that computes the value of the function in a point x=(x1,x2)
def myFunction(x):
    y = x[0]**2 * (4 - 2.1 * x[0]**2 + 1/3 * x[0]**4) + x[0]*x[1] + x[1]**2 * (-4 + 4 * x[1]**2)
    return y

# Function that computes the gradient (gx1, gx2) of the function in a point x=(x1,x2)
def myFunctionGradient(x):
    gradient = np.array([2*x[0]**5 - 8.4*x[0]**3 + 8*x[0] + x[1], 16*x[1]**3 - 8*x[1] + x[0]])
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
def gradientDescentWithDynamicAlpha(x0, alpha, function, function_gradient, max_iterations, threshold):
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
ax = fig.add_subplot(nrows, ncols, plot_number)
# Set the labels for the axes
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
# Plot filled contours (up to levels_number automatically-chosen levels)
X1 = np.linspace(INTERVAL_START_X1, INTERVAL_STOP_X1, ((INTERVAL_STOP_X1 - INTERVAL_START_X1) / SAMPLING_INTERVAL) + 1)
X2 = np.linspace(INTERVAL_START_X2, INTERVAL_STOP_X2, ((INTERVAL_STOP_X2 - INTERVAL_START_X2) / SAMPLING_INTERVAL) + 1)
X1, X2 = np.meshgrid(X1, X2)
Y = X1**2 * (4 - 2.1 * X1**2 + 1/3 * X1**4) + X1*X2 + X2**2 * (-4 + 4 * X2**2)
levels_number = 100
ax.contourf(X1, X2, Y, levels_number, cmap='inferno')

# Apply the gradient method to the function starting at the specified points
for i in range(0, COLORS.size):
    x0 = STARTING_POINTS[i]
    points = gradientDescentWithDynamicAlpha(x0, ALPHA, myFunction, myFunctionGradient, MAX_ITERATIONS[i], THRESHOLD)
    ax.plot(points[:,0], points[:,1], COLORS[i]+'o')
    for j in range(0, points.shape[0]):
        ax.annotate(j, (points[j, 0], points[j, 1]))

# Save the current figure
plt.savefig('../Images/01-gradient-descent-1st-lab-function-2nd-experiment.png')