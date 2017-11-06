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

# Create a new figure
fig = plt.figure()
# Split the figure in 2*2 (nrows*ncols) subaxes
nrows = 2
ncols = 2
for plot_number in range(1,5):
    # Get the dataset
    filename = '../Data/anscombe-dataset-' + str(plot_number) + '.csv'
    x,y = get_dataset(filename)
    # Create a new suplot positioned at plot_number
    ax = fig.add_subplot(nrows, ncols, plot_number)
    # Set the limits of the axes
    ax.set_xlim([min(x)-2,max(x)+2])
    ax.set_ylim([min(y)-2,max(y)+2])
    # Plot the points of the dataset
    ax.plot(x,y, 'ro')
# Save the current figure
plt.savefig('../Images/01-visual-analysis.png')