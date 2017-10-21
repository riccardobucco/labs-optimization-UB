import numpy as np
import scipy as sp
import scipy.optimize
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
THRESHOLDS = np.array([0.001, 0.00001, 0.0000001])
THRESHOLDS_NAMES = np.array(['001', '00001', '0000001'])
POINT_NAMES = np.array(['A', 'B'])
STARTING_POINTS = np.array([[-1.5,-0.5], [0.5, 2.5]])

# function evaluation
def func(x):
    return (A - x[0])**2 + B*(x[1] - x[0]**2)**2
    
# first order derivatives of the function
def dfunc(x):
    return np.array([2*x[0] - 2*A + B*4*x[0]**3 - 4*B*x[1]*x[0], 2*B*x[1] - 2*B*x[0]**2])
    
def gradient_descent(x, threshold):
    points = [x]
    niter = 0
    while niter <= 10000:  # maximum number of iterations
        pk = - dfunc(x)   # Search direction (observe the minus sign)
        return_values = sp.optimize.line_search(func, dfunc, x, pk)
        alpha = return_values[0]
        temp = x+alpha*pk # Observe the plus sign 
        if np.abs(func(temp)-func(x))>threshold: # convergence condition
            x = temp
            points.append(x)
        else:
            break
        niter += 1
    return x, niter, np.array(points)

X1 = np.linspace(INTERVAL_START_X1, INTERVAL_STOP_X1, ((INTERVAL_STOP_X1 - INTERVAL_START_X1) / SAMPLING_INTERVAL) + 1)
X2 = np.linspace(INTERVAL_START_X2, INTERVAL_STOP_X2, ((INTERVAL_STOP_X2 - INTERVAL_START_X2) / SAMPLING_INTERVAL) + 1)
X1, X2 = np.meshgrid(X1, X2)
Y = (A - X1)**2 + B*(X2 - X1**2)**2

for point_index, starting_point in enumerate(STARTING_POINTS):
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
        val, iters, points = gradient_descent(starting_point, threshold)
        print(threshold)
        print ("x = ", val)
        print ("iterations = ", iters)
        ax.plot(points[:,0], points[:,1], 'g')
        # Save the current figure
        plt.savefig('../Images/01-rosenbrock-function-contours-line-search-point'+POINT_NAMES[point_index]+'-threshold-'+THRESHOLDS_NAMES[threshold_index]+'.png')