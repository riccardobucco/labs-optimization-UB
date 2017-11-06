import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def get_dataset(filename):
    csv_reader = csv.DictReader(open(filename))
    x = []
    y = []
    for instance in csv_reader:
        x.append(float(instance['x']))
        y.append(float(instance['y']))
    return x,y

def myFunctionGradient(b0, b1, x, y):
    gradient_b0 = 0
    for i in range(0,len(x)):
        gradient_b0 += b0 + b1*x[i] - y[i]
    gradient_b1 = 0
    for i in range(0,len(x)):
        gradient_b1 += (b0 + b1*x[i] - y[i]) * x[i]
    return np.array([gradient_b0, gradient_b1])

# Function that implements the gradient descent method.
# It requires a starting point, alpha, a function that computes the value of a function in a specified point,
# a function that computes the gradient of a function in a specified point, the number of iterations to perform
# It returns an array of points, where each point is computed using the gradient descent method.
def gradientDescent(b0, alpha, function_gradient, iterations, x, y):
    points = [b0]
    bk = b0
    gradient_bk = function_gradient(bk[0], bk[1], x, y)
    for i in range(0, iterations):
        bk = bk - alpha*gradient_bk
        gradient_bk = function_gradient(bk[0], bk[1], x, y)
        points.append(bk)
    return np.array(points)

x = {}
y = {}

# Get the datasets
for plot_number in range(1,5):
    filename = '../Data/anscombe-dataset-' + str(plot_number) + '.csv'
    x[plot_number],y[plot_number] = get_dataset(filename)

# Start the visual analysis of the given datasets
# Create a new figure
fig = plt.figure()
# Split the figure in 2*2 (nrows*ncols) subaxes
nrows = 2
ncols = 2
for plot_number in range(1,5):
    # Create a new suplot positioned at plot_number
    ax = fig.add_subplot(nrows, ncols, plot_number)
    # Set the limits of the axes
    ax.set_xlim([min(x[plot_number])-2,max(x[plot_number])+2])
    ax.set_ylim([min(y[plot_number])-2,max(y[plot_number])+2])
    # Plot the points of the dataset
    ax.plot(x[plot_number],y[plot_number], 'bo')
# Save the current figure
plt.savefig('../Images/01-visual-analysis.png')

# Apply the matrix algebra to solve the least square problems in the given datasets
# Create a new figure
fig = plt.figure()
# Split the figure in 2*2 (nrows*ncols) subaxes
nrows = 2
ncols = 2
for plot_number in range(1,5):
     # Create a new suplot positioned at plot_number
    ax = fig.add_subplot(nrows, ncols, plot_number)
    # Set the limits of the axes
    ax.set_xlim([min(x[plot_number])-2,max(x[plot_number])+2])
    ax.set_ylim([min(y[plot_number])-2,max(y[plot_number])+2])
    # Plot the points of the dataset
    ax.plot(x[plot_number],y[plot_number], 'bo')
    # Find the solution to the least squares problem
    A = np.array([[1, xi] for xi in x[plot_number]])
    Y = np.array([[yi] for yi in y[plot_number]])
    b = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A)), Y)
    # Plot the least squares line
    points_x = np.array([min(x[plot_number])-2, max(x[plot_number])+2])
    points_y = b[0] + points_x * b[1]
    ax.plot(points_x, points_y, 'r')
# Save the current figure
plt.savefig('../Images/01-matrix algebra.png')

# Apply the gradient descent method to solve the least square problems in the given datasets
INTERVAL_START_B0 = -10
INTERVAL_STOP_B0 = 10
INTERVAL_START_B1 = -10
INTERVAL_STOP_B1 = 10
SAMPLING_INTERVAL = 0.1
STARTING_POINTS = np.array([[7,8], [9,-2], [-4,8], [-6, -7]])
ITERATIONS = 10000
ALPHA = 0.001
# Create a new figure
fig = plt.figure()
# Split the figure in 2*2 (nrows*ncols) subaxes
nrows = 2
ncols = 2
for plot_number in range(1,5):
     # Create a new suplot positioned at plot_number
    ax = fig.add_subplot(nrows, ncols, plot_number)
    B0 = np.linspace(INTERVAL_START_B0, INTERVAL_STOP_B0, ((INTERVAL_STOP_B0 - INTERVAL_START_B0) / SAMPLING_INTERVAL) + 1)
    B1 = np.linspace(INTERVAL_START_B1, INTERVAL_STOP_B1, ((INTERVAL_STOP_B1 - INTERVAL_START_B1) / SAMPLING_INTERVAL) + 1)
    B0, B1 = np.meshgrid(B0, B1)
    Q = 0
    for i in range(0,len(x[plot_number])):
        Q += (B0 + B1*x[plot_number][i] - y[plot_number][i])**2
    Q = 0.5 * Q
    levels_number = 100
    ax.contourf(B0, B1, Q, levels_number, cmap='inferno')
    points = gradientDescent(STARTING_POINTS[plot_number-1], ALPHA, myFunctionGradient, ITERATIONS, x[plot_number], y[plot_number])
    ax.plot(points[:,0], points[:,1], 'r')
    ax.plot(STARTING_POINTS[plot_number-1][0], STARTING_POINTS[plot_number-1][1], 'ro')
    ax.plot(33001.0/11000.0, 5501.0/11000.0, 'go')
    print("Dataset " + str(plot_number) + ':')
    print("beta_0 = " + str(points[-1][0]))
    print("beta_0 = " + str(points[-1][1]))
# Save the current figure
plt.savefig('../Images/01-gradient-descent-functions-to-minimize.png')