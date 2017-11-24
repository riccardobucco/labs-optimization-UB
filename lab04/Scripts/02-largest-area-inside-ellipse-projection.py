import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

INTERVAL_START_X1 = -15
INTERVAL_STOP_X1 = 15
INTERVAL_START_X2 = -60
INTERVAL_STOP_X2 = 60
SAMPLING_INTERVAL = 0.1

# This is the ellipse
a = 8.0
b = 50.0
# Tolerance
tol = 1e-08
# Initial alpha value (line search)
alpha = 1.0
# Initial values. DO NOT CHOOSE x = 0, y = 0
x = 1
y = 1
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
f= -4.0*x*y + lam * ellipse
cont = 0
print("Initial values:")
print("\tf = " + str(f))
print("\tx = " + str(x))
print("\ty = " + str(y))
points_x = []
points_y = []
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
print("\tf = " + str(f))
print("\tx = " + str(x))
print("\ty = " + str(y))
print("\tlambda = " + str(lam))
print("Correct values:")
print("\tx = " + str(np.sqrt(a**2.0 / 2.0)))
print("\ty = " + str(np.sqrt(b**2.0 / 2.0)))

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
X1_ellipse = np.linspace(-a, a, ((a - (-a)) / SAMPLING_INTERVAL) + 1)
X2_ellipse_1 = np.sqrt(b**2 * (1 - ((X1_ellipse**2)/(a**2))))
X2_ellipse_2 = -np.sqrt(b**2 * (1 - (X1_ellipse**2/a**2)))
X1 = np.linspace(INTERVAL_START_X1, INTERVAL_STOP_X1, ((INTERVAL_STOP_X1 - INTERVAL_START_X1) / SAMPLING_INTERVAL) + 1)
X2 = np.linspace(INTERVAL_START_X2, INTERVAL_STOP_X2, ((INTERVAL_STOP_X2 - INTERVAL_START_X2) / SAMPLING_INTERVAL) + 1)
X1, X2 = np.meshgrid(X1, X2)
Y = 4 * X1 * X2
levels_number = 50
ax.contourf(X1, X2, Y, levels_number, cmap='inferno')
# Plot a ellipse representing the constraint
ax.plot(X1_ellipse, X2_ellipse_1, 'r')
ax.plot(X1_ellipse, X2_ellipse_2, 'r')
# Plot the points found during the iterations
ax.plot(points_x, points_y, 'b')
# Plot the point that minimized the function (analitycally found)
ax.plot(np.sqrt(a**2/2), np.sqrt(b**2/2), 'go')
# Save the current figure
plt.savefig('../Images/02-projection.png')