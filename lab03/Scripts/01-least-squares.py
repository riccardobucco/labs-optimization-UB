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
    print(points_x)
    print(points_y)
    ax.plot(points_x, points_y, 'r')
# Save the current figure
plt.savefig('../Images/01-matrix algebra.png')