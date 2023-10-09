import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

sigma = 0.1
centers = []
gauss_count = 25
loop_size = 5
w_list = [.5] * (gauss_count + 1)


def gaussian(x_val, y_val, center, sigma_error):
    return np.exp(
        -((x_val - center[0]) ** 2 / (2 * sigma_error ** 2) + (y_val - center[1]) ** 2 / (2 * sigma_error ** 2)))


def model_simple(xy, *amplitudes):
    x, y = xy
    h = amplitudes[0]
    for i in range(1, len(amplitudes)):
        h += amplitudes[i] * gaussian(x, y, centers[i - 1], sigma)
    return h


if __name__ == '__main__':
    # load data
    data_Dayan: np.array = np.loadtxt('data_3d.csv', delimiter=',').T

    X_Dayan = data_Dayan[0]
    Y_Dayan = data_Dayan[1]
    Z_Dayan = data_Dayan[2]

    step_size = 1 / loop_size
    vertex_pos = step_size / 2
    temp = vertex_pos

    for i in range(0, loop_size):
        for j in range(0, loop_size):
            centers.append([temp, vertex_pos + (step_size * j)])
        temp += step_size

    # Perform curve fitting
    w_opt, _ = curve_fit(model_simple, (X_Dayan, Y_Dayan), Z_Dayan, p0=w_list)

    # Print optimized parameters
    print(w_opt)

    # Create 3D plot of the data points and the fitted curve
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_Dayan, Y_Dayan, Z_Dayan, color='blue', alpha=0.1)
    x_range = np.linspace(0, 1, 50)
    y_range = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = model_simple((X, Y), *w_opt)
    ax.plot_surface(X, Y, Z, color='red', alpha=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
