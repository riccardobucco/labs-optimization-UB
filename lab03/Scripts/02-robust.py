import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


C = 1
MAX_ITERATIONS = 100000
THRESHOLD = 0.0000001

INTERVAL_START_U = -10
INTERVAL_STOP_U = 10
SAMPLING_INTERVAL = 0.1

# Get the dataset from a CSV file
def get_dataset(filename):
    csv_reader = csv.DictReader(open(filename))
    x = []
    y = []
    for instance in csv_reader:
        x.append(float(instance['x']))
        y.append(float(instance['y']))
    return x,y

def cauchy(b, xi, yi, C):
    u = b[0] + b[1]*xi - yi
    return C**2 * 0.5 * np.log(1 + (u / C)**2)

def cauchyGrad(b, xi, yi, C):
    u = b[0] + b[1]*xi - yi
    return np.array([(C**2 * u) / (1 + u**2), (xi * C**2 * u) / (1 + u**2)])

def huber(b, xi, yi, C):
    u = b[0] + b[1]*xi - yi
    if abs(u) <= C:
        return 0.5 * u**2
    else:
        return 0.5 * C * (2 * abs(u) - C)

def huberGrad(b, xi, yi, C):
    u = b[0] + b[1]*xi - yi
    if abs(u) <= C:
        return np.array([u, u * xi])
    else:
        return np.array([np.sign(u), np.sign(u) * xi])

def Q(b, x, y, C, rho):
    Q = 0
    for i in range(0, len(x)):
        Q += rho(b, x[i], y[i], C)
    return Q

def QGrad(b, x, y, C, rhoGrad):
    dQ_db0 = 0
    dQ_db1 = 0
    for i in range(0, len(x)):
        rho_grad = rhoGrad(b, x[i], y[i], C)
        dQ_db0 += rho_grad[0]
        dQ_db1 += rho_grad[1]
    return np.array([dQ_db0, dQ_db1])

# Function that finds an alpha such that f(xk - alpha*grad(f(xk))) < f(xk)
# alpha is found starting from 1 and dividing iteratively by 2 until the condition is satisfied 
def findAlpha(bk, x, y, C, function, function_gradient, rho, rho_grad):
    alpha = 1.0
    bk_new = bk - alpha*function_gradient(bk, x, y, C, rho_grad)
    while function(bk_new, x, y, C, rho) >= function(bk, x, y, C, rho):
        alpha = alpha / 2
        bk_new = bk - alpha*function_gradient(bk, x, y, C, rho_grad)
    return alpha

# Function that implements the gradient descent method, with alpha that changes at every iteration.
# It requires a starting point, alpha, a function that computes the value of a function in a specified point,
# a function that computes the gradient of a function in a specified point, the maximum number of iterations
# to perform and a threshold to stop the method.
# It returns an array of points, where each point is computed using the gradient descent method.
def gradientDescentWithDynamicAlpha(b0, x, y, C, function, function_gradient, rho, rho_grad, max_iterations, threshold):
    points = [b0]
    bk = b0
    alpha = findAlpha(bk, x, y, C, function, function_gradient, rho, rho_grad)
    bk1 = bk - alpha*function_gradient(bk, x, y, C, rho_grad)
    i = 1
    while (i < max_iterations) & (abs(function(bk1, x, y, C, rho) - function(bk, x, y, C, rho)) > threshold):
        bk = bk1
        alpha = findAlpha(bk, x, y, C, function, function_gradient, rho, rho_grad)
        bk1 = bk - alpha*function_gradient(bk, x, y, C, rho_grad)
        points.append(bk1)
        i = i + 1
    return np.array(points)

x = {}
y = {}

# Get the datasets
for i in range(1,5):
    filename = '../Data/anscombe-dataset-' + str(i) + '.csv'
    x[i], y[i] = get_dataset(filename)

# Compare different weights: OLS, Huber and Cauchy
U = np.linspace(INTERVAL_START_U, INTERVAL_STOP_U, ((INTERVAL_STOP_U - INTERVAL_START_U) / SAMPLING_INTERVAL) + 1)
# Create a new figure
fig = plt.figure()
# Compute the value of the functions
W_OLS = np.zeros(U.shape)
W_huber = np.zeros(U.shape)
W_cauchy = np.zeros(U.shape)
for i in range(0, len(U)):
    W_OLS[i] = 0.5 * U[i]**2
    if abs(U[i]) <= C:
        W_huber[i] = 0.5 * U[i]**2
    else:
        W_huber[i] = 0.5 * C * (2 * abs(U[i]) - C)
    W_cauchy[i] = C**2 * 0.5 * np.log(1 + (U[i] / C)**2)
# Create a new suplot positioned at top-left and plot the OLS weights
ax = plt.subplot2grid((2,2), (0,0))
ax.plot(U, W_OLS, 'r')
# Create a new suplot positioned at top-right and plot the OLS weights
ax = plt.subplot2grid((2,2), (0,1))
ax.plot(U, W_huber, 'b')
# Create a new suplot positioned at top-left and plot the OLS weights
ax = plt.subplot2grid((4,4), (2,1), colspan=2, rowspan=2)
ax.plot(U, W_cauchy, 'g')
# Save the current figure

plt.savefig('../Images/02-robust-weights-comparison.png')

# Focus on the third Anscombe's dataset
# Find the solution to the least squares problem
A = np.array([[1, xi] for xi in x[3]])
Y = np.array([[yi] for yi in y[3]])
b_ols = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A)), Y)

# Find the solution to the problem applying Huber and Cauchy functions, and using the solution of the OLS
# problem as a starting point
points_huber = gradientDescentWithDynamicAlpha(b_ols, x[3], y[3], C, Q, QGrad, huber, huberGrad, MAX_ITERATIONS, THRESHOLD)
points_cauchy = gradientDescentWithDynamicAlpha(b_ols, x[3], y[3], C, Q, QGrad, cauchy, cauchyGrad, MAX_ITERATIONS, THRESHOLD)
b_huber = points_huber[-1]
b_cauchy = points_cauchy[-1]

# Print the results
print("Least Squares (Dataset 3)")
print("\tb0 = " + str(b_ols[0]))
print("\tb1 = " + str(b_ols[1]))
print("Huber (Dataset 3)")
print("\tb0 = " + str(b_huber[0]))
print("\tb1 = " + str(b_huber[1]))
print("Cauchy (Dataset 3)")
print("\tb0 = " + str(b_cauchy[0]))
print("\tb1 = " + str(b_cauchy[1]))

# Create a new figure
fig = plt.figure()
# Split the figure in 1*1 (nrows*ncols) subaxes
nrows = 1
ncols = 1
# Create a new suplot positioned
ax = fig.add_subplot(nrows, ncols, 1)
# Set the limits of the axes
ax.set_xlim([min(x[3])-2,max(x[3])+2])
ax.set_ylim([min(y[3])-2,max(y[3])+2])
# Plot the points of the dataset
ax.plot(x[3],y[3], 'bo')
# Plot the least squares line
points_x = np.array([min(x[3])-2, max(x[3])+2])
points_y = b_ols[0] + points_x * b_ols[1]
ax.plot(points_x, points_y, 'r')
# Plot the Huber line
points_y = b_huber[0] + points_x * b_huber[1]
ax.plot(points_x, points_y, 'b')
# Save the current figure
plt.savefig('../Images/02-robust-huber-3rd-dataset.png')

# Create a new figure
fig = plt.figure()
# Split the figure in 1*1 (nrows*ncols) subaxes
nrows = 1
ncols = 1
# Create a new suplot positioned
ax = fig.add_subplot(nrows, ncols, 1)
# Set the limits of the axes
ax.set_xlim([min(x[3])-2,max(x[3])+2])
ax.set_ylim([min(y[3])-2,max(y[3])+2])
# Plot the points of the dataset
ax.plot(x[3],y[3], 'bo')
# Plot the least squares line
points_x = np.array([min(x[3])-2, max(x[3])+2])
points_y = b_ols[0] + points_x * b_ols[1]
ax.plot(points_x, points_y, 'r')
# Plot the Cauchy line
points_y = b_cauchy[0] + points_x * b_cauchy[1]
ax.plot(points_x, points_y, 'g')
# Save the current figure
plt.savefig('../Images/02-robust-cauchy-3rd-dataset.png')