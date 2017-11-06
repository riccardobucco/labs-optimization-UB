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
    return np.array(x), np.array(y)

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