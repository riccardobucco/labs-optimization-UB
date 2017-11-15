import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

INTERVAL_START_X1 = -15
INTERVAL_STOP_X1 = 15
INTERVAL_START_X2 = -60
INTERVAL_STOP_X2 = 60
SAMPLING_INTERVAL = 0.1

# The solution presented here maximizes the area of
# the rectangle trying to ensure that the solution
# of the corners of the rectangle are over an ellipse.
# For that issue we give a large penalization for the
# ellipse term.
# Penalty
c = 2000000.0
# This is the ellipse
a = 8.0
b = 50.0
# Tolerance
tol = 1e-8
alpha = 1.0
# Initial values
x = 1.0
y = 1.0
# Points x and y should be over the ellipse. We therefore
# include a large penalty for the restriction the solution
# should have: it has to be over the ellipse. Do not 
# interpret the next equation as the Lagrangian! 
ellipse = (x**2.0 / a**2.0 + y**2.0 / b**2.0 - 1)
f = -4.0*x*y + c/2.0 * ellipse**2.0
cont = 0
points_x = []
points_y = []
while alpha > tol and cont < 100000:
    points_x.append(x)
    points_y.append(y)
    cont += 1
    # Gradient. Point x and y should be over the ellipse.
    # The penalty is used to ensure that the solution is
    # over the ellipse
    ellipse = (x**2.0 / a**2.0 + y**2.0 / b**2.0 - 1)
    grx = -4.0*y + c * ellipse * 2.0 * x / a**2.0
    gry = -4.0*x + c * ellipse * 2.0 * y / b**2.0
    # Normalization of the gradient. Just to avoid "jumping" too
    # far away with the line search
    modulus2 = grx**2.0 + gry**2.0
    modulus  = np.sqrt(modulus2)
    grx = grx / modulus
    gry = gry / modulus
    # Used to know if we finished line search
    finished = 0
    while finished == 0 and alpha > tol:
        # Update
        aux_x = x - alpha*grx
        aux_y = y - alpha*gry
        # Compute new value of the energy
        ellipse = (aux_x**2.0 / a**2.0 + aux_y**2.0 / b**2.0 - 1)
        aux_f = -4.0*aux_x*aux_y + c/2.0 * ellipse**2.0
        # Check if this is a descent
        if aux_f < f:
            x = aux_x
            y = aux_y
            f = aux_f
            alpha = 1.0
            finished = 1
        else:
            alpha = alpha/2.0
print("Number of iterations: " + str(cont))
print("Final values:")
print("\tx = " + str(x))
print("\ty = " + str(y))
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
plt.savefig('../Images/02-penalization.png')