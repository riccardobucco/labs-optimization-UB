import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

INTERVAL_START_X1 = -5
INTERVAL_STOP_X1 = 5
INTERVAL_START_X2 = -50
INTERVAL_STOP_X2 = 50
SAMPLING_INTERVAL = 0.1
THRESHOLD = 0.001
MAX_ITERATIONS = 1000
STARTING_POINTS = np.array([[4, 40], [-2, -20]])
COLORS = np.array(['g', 'y'])

# Function that computes the value of the function in a point x=(x1,x2)
def myFunction(x):
    y = 100*x[0,0]**2 + x[1,0]**2
    return y

# Function that computes the gradient (gx1, gx2) of the function in a point x=(x1,x2)^T
def myFunctionGradient(x):
    gradient = np.array([[200*x[0,0]], [2*x[1,0]]])
    return gradient

# Function that computes the Hessian of the function in a point x=(x1,x2)^T
def myFunctionHessian(x):
    hessian = np.array([[200, 0], [0, 2]])
    return hessian

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
    xk = np.array([[x0[0]], [x0[1]]])
    alpha = findAlpha(xk, function, function_gradient)
    xk1 = xk - alpha*function_gradient(xk)
    i = 1
    while (i < max_iterations) & (abs(function(xk1) - function(xk)) > threshold):
        xk = xk1
        alpha = findAlpha(xk, function, function_gradient)
        xk1 = xk - alpha*function_gradient(xk)
        points.append(np.array([xk1[0,0], xk1[1,0]]))
        i = i + 1
    return np.array(points)

def newtonDescentWithDynamicAlpha(x0, function, function_gradient, function_hessian, max_iterations, threshold):
    points = [x0]
    xk = np.array([[x0[0]], [x0[1]]])
    alpha = findAlpha(xk, function, function_gradient)
    xk1 = xk + np.dot(alpha*np.linalg.inv(function_hessian(xk)), (-function_gradient(xk)))
    i = 1
    while (i < max_iterations) & (abs(function(xk1) - function(xk)) > threshold):
        xk = xk1
        alpha = findAlpha(xk, function, function_gradient)
        xk1 = xk + np.dot(alpha*np.linalg.inv(function_hessian(xk)), (-function_gradient(xk)))
        points.append(np.array([xk1[0,0], xk1[1,0]]))
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
Y = 100*X1**2 + X2**2
levels_number = 100
ax.contourf(X1, X2, Y, levels_number, cmap='inferno')
# Save the current figure
plt.savefig('../Images/02-simple-quadratic-function-countours.png')

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
Y = 100*X1**2 + X2**2
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='inferno')
# Save the current figure
plt.savefig('../Images/02-simple-quadratic-function-surface.png')

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
ax.plot(0, 0, 'ro')
# Compute how many steps you need to perform using the backtracking algorithm to arrive
# under a specified threshold
for i in range(0, STARTING_POINTS.shape[0]):
    starting_point = STARTING_POINTS[i]
    points_backtracking = gradientDescentWithDynamicAlpha(starting_point, myFunction, myFunctionGradient, MAX_ITERATIONS, THRESHOLD)
    print(points_backtracking.shape[0])
    ax.plot(points_backtracking[:,0], points_backtracking[:,1], COLORS[i])
# Save the current figure
plt.savefig('../Images/02-simple-quadratic-function-backtracking.png')

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
ax.plot(0, 0, 'ro')
# Compute how many steps you need to perform using the Newton algorithm to arrive
# under a specified threshold
for i in range(0, STARTING_POINTS.shape[0]):
    starting_point = STARTING_POINTS[i]
    points_newton = newtonDescentWithDynamicAlpha(starting_point, myFunction, myFunctionGradient, myFunctionHessian, MAX_ITERATIONS, THRESHOLD)
    print(points_newton.shape[0])
    ax.plot(points_newton[:,0], points_newton[:,1], COLORS[i])
# Save the current figure
plt.savefig('../Images/02-simple-quadratic-function-newton.png')