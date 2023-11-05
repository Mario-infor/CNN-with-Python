import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Global parameters and hyperparameters.
sigma = 0.5
centers = []
loop_size = 6
gauss_count = loop_size * loop_size
w_list = [.5] * (gauss_count + 1)


# Evaluate a coordinate (x, y) on one gaussian curve.
def gaussian(x_val, y_val, center, sigma_error):
    return np.exp(
        -((x_val - center[0]) ** 2 / (2 * sigma_error ** 2) + (y_val - center[1]) ** 2 / (2 * sigma_error ** 2)))


# Evaluate coordinate (x, y) on all gaussian curves on the surface.
def model(xy, *amplitudes):
    _x, _y = xy
    h = amplitudes[0]
    for _i in range(1, len(amplitudes)):
        h += amplitudes[_i] * gaussian(x, y, centers[_i - 1], sigma)
    return h


if __name__ == '__main__':
    # load data
    data: np.array = np.loadtxt('data_3d.csv', delimiter=',').T

    x = data[0]
    y = data[1]
    z = data[2]

    # Create centers for the gaussian surface.
    step_size = 1 / loop_size
    vertex_pos = step_size / 2
    temp = vertex_pos

    for i in range(0, loop_size):
        for j in range(0, loop_size):
            centers.append([temp, vertex_pos + (step_size * j)])
        temp += step_size

    # Perform curve fitting
    w_opt, _ = curve_fit(model, (x, y), z, p0=w_list)

    # Print optimized parameters
    print(w_opt)

    # Create 3D plot of the data points and the fitted curve
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color='blue', alpha=0.1)
    x_range = np.linspace(0, 1, 50)
    y_range = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = model((X, Y), *w_opt)
    ax.plot_surface(X, Y, Z, color='red', alpha=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
