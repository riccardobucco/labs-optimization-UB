import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

INTERVAL_START_X1 = -2
INTERVAL_STOP_X1 = 2
INTERVAL_START_X2 = -2
INTERVAL_STOP_X2 = 2
SAMPLING_INTERVAL = 0.01

# This is the ellipse
a = 1.0
b = 1.0
# Tolerance
tol = 1e-08
# Initial alpha value (line search)
alpha = 1.0

# Set the seed (so that you can repeat the experiment)
np.random.seed(seed=25101995)

# Get random samples from a multivariate normal distribution
m1 = [4.,-1.]
s1 = [[1,0.9],[0.9,1]]
c1 = np.random.multivariate_normal(m1,s1,100)
# Covariance matrix of tha data
A = np.cov(c1.T)
# Find eigenvector of maximum eigenvalue
vaps,veps = np.linalg.eig(np.cov(c1.T))
idx = np.argmax(vaps)

# Initial values. DO NOT CHOOSE x = 0, y = 0
x = 0.3
y = 0.5
points_x = []
points_y = []
points_x.append(x)
points_y.append(y)
# Projection of (x,y) over the ellipse. In order to compute the
# projection we compute the intersection of the line passing
# through the origin (0,0) and the point (x,y) with the ellipse
# x^2/a^2+y^2/b^2=1. 
den = np.sqrt(a**2.0 * y**2.0 + b**2.0 * x**2.0)
x = a * b * x / den
y = a * b * y / den
# Given current values of  (x,y), compute the value of lambda
# that minimizes the modulus of the gradient of the Lagrangian
lam = 2.0 * a**2.0 * b**2.0 * (b**2.0 + a**2.0) * x * y / (a**4.0 * y**2.0 + b**4.0 * x**2.0)
# Compute Lagrangian. Points x and y should be over the ellipse
ellipse = x**2.0 / a**2.0 + y**2.0 / b**2.0 - 1
f = -4.0*x*y + lam * ellipse
cont = 0
while alpha > tol and cont < 100000:
    points_x.append(x)
    points_y.append(y)
    cont = cont+1
    # Gradient of the Lagrangian
    grx = -4.0*y + 2.0*lam*x/a**2.0
    gry = -4.0*x + 2.0*lam*y/b**2.0
    # Used to know if we finished line search
    finished = 0
    while finished == 0 and alpha > tol:
        # Update
        aux_x = x - alpha*grx
        aux_y = y - alpha*gry
        # Projection of (aux_x, aux_y) over the ellipse. This is done
        # as explained before
        den = np.sqrt(a**2.0 * aux_y**2.0 + b**2.0 * aux_x**2.0)
        aux_x = a * b * aux_x / den
        aux_y = a * b * aux_y / den
        # Compute new value of the Lagrangian. 
        ellipse = aux_x**2.0 / a**2.0 + aux_y**2.0 / b**2.0 - 1
        aux_lam = 2.0 * a**2.0 * b**2.0 * (b**2.0 + a**2.0) * aux_x * aux_y / (a**4.0 * aux_y**2.0 + b**4.0 * aux_x**2.0)
        aux_f = -4.0*aux_x*aux_y + aux_lam*ellipse
        # Check if this is a descent
        if aux_f < f:
            x = aux_x
            y = aux_y
            lam = aux_lam
            f = aux_f
            alpha = 1.0
            finished = 1
        else:
            alpha=alpha/2.0
print("Number of iterations: " + str(cont))
print("Final values:")
print("\tx = " + str(x))
print("\ty = " + str(y))
print("Correct values:")
print("\tx = " + str(veps[0,idx]))
print("\ty = " + str(veps[1,idx]))

# Create a new figure
fig = plt.figure()
# Split the figure in 1*1 (nrows*ncols) subaxes and create a new suplot positioned at 1 (plot_number)
nrows = 1
ncols = 1
plot_number = 1
ax = fig.add_subplot(nrows, ncols, plot_number)
# Set the labels for the axes
ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
# Plot filled contours (up to levels_number automatically-chosen levels)
X1_ellipse = np.linspace(-a, a, ((a - (-a)) / SAMPLING_INTERVAL) + 1)
X2_ellipse_1 = np.sqrt(b**2 * (1 - ((X1_ellipse**2)/(a**2))))
X2_ellipse_2 = -np.sqrt(b**2 * (1 - (X1_ellipse**2/a**2)))
X1 = np.linspace(INTERVAL_START_X1, INTERVAL_STOP_X1, ((INTERVAL_STOP_X1 - INTERVAL_START_X1) / SAMPLING_INTERVAL) + 1)
X2 = np.linspace(INTERVAL_START_X2, INTERVAL_STOP_X2, ((INTERVAL_STOP_X2 - INTERVAL_START_X2) / SAMPLING_INTERVAL) + 1)
X1, X2 = np.meshgrid(X1, X2)
Y = np.zeros([len(X1), len(X1)])
for i in range(0,len(X1)):
    for j in range(0,len(X1)):
        Y[i,j] = (X1[i,j]*A[0,0] + X2[i,j]*A[1,0])*X1[i,j] + (X1[i,j]*A[0,1] + X2[i,j]*A[1,1])*X2[i,j]
levels_number = 100
ax.contourf(X1, X2, Y, levels_number, cmap='inferno')
# Plot a ellipse representing the constraint
ax.plot(X1_ellipse, X2_ellipse_1, 'r')
ax.plot(X1_ellipse, X2_ellipse_2, 'r')
# Plot the point that minimized the function (analitycally found)
ax.plot(veps[0,idx], veps[1,idx], 'go')
# Save the current figure
plt.savefig('../Images/03-pca-analysis-analytical.png')

# Create a new figure
fig = plt.figure()
# Split the figure in 1*1 (nrows*ncols) subaxes and create a new suplot positioned at 1 (plot_number)
nrows = 1
ncols = 1
plot_number = 1
ax = fig.add_subplot(nrows, ncols, plot_number)
# Set the labels for the axes
ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
# Plot filled contours (up to levels_number automatically-chosen levels)
X1_ellipse = np.linspace(-a, a, ((a - (-a)) / SAMPLING_INTERVAL) + 1)
X2_ellipse_1 = np.sqrt(b**2 * (1 - ((X1_ellipse**2)/(a**2))))
X2_ellipse_2 = -np.sqrt(b**2 * (1 - (X1_ellipse**2/a**2)))
X1 = np.linspace(INTERVAL_START_X1, INTERVAL_STOP_X1, ((INTERVAL_STOP_X1 - INTERVAL_START_X1) / SAMPLING_INTERVAL) + 1)
X2 = np.linspace(INTERVAL_START_X2, INTERVAL_STOP_X2, ((INTERVAL_STOP_X2 - INTERVAL_START_X2) / SAMPLING_INTERVAL) + 1)
X1, X2 = np.meshgrid(X1, X2)
Y = np.zeros([len(X1), len(X1)])
for i in range(0,len(X1)):
    for j in range(0,len(X1)):
        Y[i,j] = (X1[i,j]*A[0,0] + X2[i,j]*A[1,0])*X1[i,j] + (X1[i,j]*A[0,1] + X2[i,j]*A[1,1])*X2[i,j]
        Y[i,j] = -Y[i,j]
levels_number = 100
ax.contourf(X1, X2, Y, levels_number, cmap='inferno')
# Plot a ellipse representing the constraint
ax.plot(X1_ellipse, X2_ellipse_1, 'r')
ax.plot(X1_ellipse, X2_ellipse_2, 'r')
# Plot the points found during the iterations
ax.plot(points_x, points_y, 'b')
# Plot the point that minimized the function (analitycally found)
ax.plot(veps[0,idx], veps[1,idx], 'go')
# Save the current figure
plt.savefig('../Images/03-pca-analysis-numerical.png')

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
# Plot the samples
ax.plot(c1[:,0],c1[:,1],'r.')
# Plot the eigenvector related to the maximum eigenvalue
plt.arrow(np.mean(c1[:,0]),np.mean(c1[:,1]),
    vaps[idx]*veps[0,idx],vaps[idx]*veps[1,idx],0.5,
    linewidth=1,head_width=0.1,color='blue')
# Save the current figure
plt.savefig('../Images/03-pca-analysis-scatterplot.png')