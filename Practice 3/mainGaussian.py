import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def func(xy, a, b, c, d, e, f):
    x, y = xy
    return a + b * x + c * y + d * x ** 2 + e * y ** 2 + f * x * y


if __name__ == '__main__':
    # load data
    data_Dayan: np.array = np.loadtxt('data_3d.csv', delimiter=',').T

    # Generate random 3D data points
    x = np.random.random(100)
    y = np.random.random(100)
    z = np.sin(x * y) + np.random.normal(0, 0.1, size=100)
    data = np.array([x, y, z]).T

    X_Dayan = data_Dayan[0]
    Y_Dayan = data_Dayan[1]
    Z_Dayan = data_Dayan[2]

    # Perform curve fitting
    popt, pcov = curve_fit(func,(X_Dayan, Y_Dayan), Z_Dayan)

    # Print optimized parameters
    print(popt)

    # Create 3D plot of the data points and the fitted curve
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_Dayan, Y_Dayan, Z_Dayan, color='blue')
    x_range = np.linspace(0, 1, 50)
    y_range = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func((X, Y), *popt)
    ax.plot_surface(X, Y, Z, color='red', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
