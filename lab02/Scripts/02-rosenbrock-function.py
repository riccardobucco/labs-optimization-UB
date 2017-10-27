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
THRESHOLD = 0.001
STARTING_POINT = np.array([-1.5,-0.5])

# Function that computes the value of the function in a point x=(x1,x2)
def myFunction(x):
    y = (A - x[0])**2 + B*(x[1] - x[0]**2)**2
    return y

# Function that computes the gradient (gx1, gx2) of the function in a point x=(x1,x2)
def myFunctionGradient(x):
    gradient = np.array([2*x[0] - 2*A + B*4*x[0]**3 - 4*B*x[1]*x[0], 2*B*x[1] - 2*B*x[0]**2])
    return gradient

# Function that computes the Hessian of the function in a point x=(x1,x2)^T
def myFunctionHessian(x):
    hessian = np.array([[2 + 12*B*x[0,0]**2 - 4*B*x[1,0], -4*B*x[0,0]], [-4*B*x[0,0], 2*B]])
    return hessian

# Function that finds an alpha such that f(xk - alpha*grad(f(xk))) < f(xk)
# alpha is found starting from 1 and dividing iteratively by 2 until the condition is satisfied 
def findAlpha(xk, function, function_gradient):
    alpha = 1.0
    while function(xk - alpha*function_gradient(xk)) >= function(xk):
        alpha = alpha / 2
    return alpha

# Function that finds an alpha such that f(xk + alpha*(hessian(f(xk)))^(-1)*(-grad(f(xk)))) <= f(xk)
# alpha is found starting from 1 and dividing iteratively by 2 until the condition is satisfied 
def findAlphaNewton(xk, function, function_gradient, function_hessian):
    alpha = 1.0
    while function(xk + np.dot(alpha*np.linalg.inv(function_hessian(xk)), (-function_gradient(xk)))) > function(xk):
        alpha = alpha / 2
    return alpha

def isPositiveDefinite(x):
    return np.all(np.linalg.eigvals(x) > 0)

def newtonGradientDescentWithDynamicAlpha(x0, function, function_gradient, function_hessian, max_iterations, threshold):
    points = [x0]
    colors = []
    xk = np.array([[x0[0]], [x0[1]]])
    xk1 = xk
    alpha = 1
    if isPositiveDefinite(function_hessian(xk)):
        alpha = findAlphaNewton(xk, function, function_gradient, function_hessian)
        xk1 = xk + np.dot(alpha*np.linalg.inv(function_hessian(xk)), (-function_gradient(xk)))
        colors.append('r')
    else:
        alpha = findAlpha(xk, function, function_gradient)
        xk1 = xk - alpha*function_gradient(xk)
        colors.append('g')
    i = 1
    while (i < max_iterations) & (abs(function(xk1) - function(xk)) > threshold):
        xk = xk1
        if isPositiveDefinite(function_hessian(xk)):
            alpha = findAlphaNewton(xk, function, function_gradient, function_hessian)
            xk1 = xk + np.dot(alpha*np.linalg.inv(function_hessian(xk)), (-function_gradient(xk)))
            colors.append('g')
        else:
            alpha = findAlpha(xk, function, function_gradient)
            xk1 = xk - alpha*function_gradient(xk)
            colors.append('r')
        points.append(np.array([xk1[0,0], xk1[1,0]]))
        i = i + 1
    return np.array(points), np.array(colors)

X1 = np.linspace(INTERVAL_START_X1, INTERVAL_STOP_X1, ((INTERVAL_STOP_X1 - INTERVAL_START_X1) / SAMPLING_INTERVAL) + 1)
X2 = np.linspace(INTERVAL_START_X2, INTERVAL_STOP_X2, ((INTERVAL_STOP_X2 - INTERVAL_START_X2) / SAMPLING_INTERVAL) + 1)
X1, X2 = np.meshgrid(X1, X2)
Y = (A - X1)**2 + B*(X2 - X1**2)**2

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
# Apply the Newton/Gradient descent method to the starting point
points, colors = newtonGradientDescentWithDynamicAlpha(STARTING_POINT, myFunction, myFunctionGradient, myFunctionHessian, MAX_ITERATIONS, THRESHOLD)
for j in range(0, points.shape[0]):
    ax.plot(points[j,0], points[j,1], colors[j]+'o')
# Plot the global minimum of the function
ax.plot(A, A**2, 'yo')
# Save the current figure
plt.savefig('../Images/02-rosenbrock-function.png')