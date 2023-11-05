from random import random
from random import seed
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# Global parameters and hyperparameters.
centers = []
sigma = 1.2
alpha = 0.005
iterations = 2000
gauss_count = 25
loop_size = 5
w_list = [0] * (gauss_count + 1)
TOTAL_NUM = 500


# Calculate error of the surface.
def error(_x, _y):
    _g = calculate_g(_x, _y)
    if _g >= 0.5:
        return _g, "Red"
    else:
        return _g, "Blue"


# Evaluate a coordinate (x, y) on one gaussian curve.
def gaussian(x_val, y_val, center, _sigma):
    return np.exp(
        -((x_val - center[0]) ** 2 / (2 * _sigma ** 2) + (y_val - center[1]) ** 2 / (2 * _sigma ** 2)))


# Evaluate coordinate (x, y) on all gaussian curves on the surface.
def calculate_g(_x, _y):
    h = w_list[0]
    for _i in range(1, len(w_list)):
        h += w_list[_i] * gaussian(_x, _y, centers[_i - 1], sigma)
    _g = 1 / (1 + np.exp(-h))
    return g


if __name__ == '__main__':
    # seed random number generator
    seed(1)

    X = []
    Y = []

    x1 = []
    y1 = []

    x2 = []
    y2 = []

    # Generate random values between [-10; 10].
    for _ in range(TOTAL_NUM):
        X.append(random() * 20 - 10)
        Y.append(random() * 20 - 10)

    # Separate points to be in the center of the graph or not.
    for i in range(TOTAL_NUM):
        dist = np.sqrt(X[i] * X[i] + Y[i] * Y[i])
        if dist < 4:
            x1.append(X[i])
            y1.append(Y[i])
        elif dist > 6:
            x2.append(X[i])
            y2.append(Y[i])

    # Create centers for the gaussian surface.
    step_size = (max(X) - min(X)) / loop_size
    vertex_initial_pos = min(X) + step_size / 2
    temp = vertex_initial_pos
    for i in range(0, loop_size):
        for j in range(0, loop_size):
            centers.append([temp, vertex_initial_pos + (step_size * j)])
        temp += step_size

    # Create mesh and draw initial surface shape.
    x = np.linspace(min(X), max(X), 100)
    y = np.linspace(min(Y), max(Y), 100)
    X_mesh, Y_mesh = np.meshgrid(x, y)

    g = calculate_g(X_mesh, Y_mesh)

    z1 = np.full(len(x1), 0.9)
    z2 = np.full(len(x2), 1)
    z = np.concatenate((z1, z2))

    trace1 = go.Scatter3d(x=x1 + x2, y=y1 + y2, z=z, mode='markers', marker=dict(size=5, color=z))
    trace2 = go.Surface(z=g, x=X_mesh, y=Y_mesh, colorscale='Viridis', showscale=False, opacity=0.5)

    fig = go.Figure(data=[trace1, trace2])
    fig.show()

    # Train weights for the gaussian surface.
    train_error_list = []
    for ite in range(iterations):
        for i in range(len(x1)):
            g = calculate_g(x1[i], y1[i])

            for j in range(1, len(w_list)):
                w_list[j] = w_list[j] + alpha * (1 - g) * gaussian(x1[i], y1[i], centers[j - 1], sigma)

        train_error = 0

        # Calculate error base on current weights of gaussian surface.
        for i in range(len(x1)):
            g, tag = error(x1[i], y1[i])
            error_value = (g - 1) ** 2
            train_error += error_value

        train_error_list.append(train_error)

        # Draw graph every 100 iterations.
        if ite % 100 == 0:
            g = calculate_g(X_mesh, Y_mesh)

            trace2 = go.Surface(z=g, x=X_mesh, y=Y_mesh, colorscale='Viridis', showscale=False, opacity=0.5)
            fig = go.Figure(data=[trace1, trace2])
            fig.show()

    print(w_list)
    plt.plot(train_error_list)
