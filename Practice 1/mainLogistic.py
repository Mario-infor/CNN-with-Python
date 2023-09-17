from random import random
from random import seed

import matplotlib.pyplot as plt
import numpy as np


def error(x, y):
    H = w0 + w1 * x + w2 * x + w3 * (x * y) + w4 * (x ** 2) + w5 * (x ** 2)
    G = 1 / (1 + np.power(np.exp(1), (-1 * H)))
    if G >= 0.5:
        return G, "Rojo"
    else:
        return G, "Azul"


if __name__ == '__main__':
    # seed random number generator
    seed(1)

    x1 = []
    y1 = []

    x2 = []
    y2 = []

    TOTAL_NUM = 500

    alpha = 0.0009
    iterations = 100

    w0 = random() - 0.5
    w1 = random() - 0.5
    w2 = random() - 0.5
    w3 = random() - 0.5
    w4 = random() - 0.5
    w5 = random() - 0.5

    # Plotting points
    plt.plot(x1, y1, 'ro')
    plt.plot(x2, y2, 'bo')
    # Plotting model
    x = np.linspace(-10.0, 10.0, 100)
    y = np.linspace(-10.0, 10.0, 100)
    X, Y = np.meshgrid(x, y)
    F = w0 + w1 * X + w2 * Y + w3 * (X * Y) + w4 * (X ** 2) + w5 * (Y ** 2)
    plt.contour(X, Y, F, [0])
    # Plotting all
    plt.show()

    train_error_list = []
    for ite in range(iterations):
        # Positive examples
        for i in range(len(x1)):
            H = w0 + w1 * x1[i] + w2 * y1[i] + w3 * (x1[i] * y1[i]) + w4 * (x1[i] ** 2) + w5 * (y1[i] ** 2)
            G = 1 / (1 + np.power(np.exp(1), (-1 * H)))
            w0 = w0 + alpha * (1 - G)
            w1 = w1 + alpha * (1 - G) * (x1[i])
            w2 = w2 + alpha * (1 - G) * (y1[i])
            w3 = w3 + alpha * (1 - G) * (x1[i] * y1[i])
            w4 = w4 + alpha * (1 - G) * (x1[i] ** 2)
            w5 = w5 + alpha * (1 - G) * (y1[i] ** 2)
        # Negative examples
        for i in range(len(x2)):
            H = w0 + w1 * x2[i] + w2 * y2[i] + w3 * (x2[i] * y2[i]) + w4 * (x2[i] ** 2) + w5 * (y2[i] ** 2)
            G = 1 / (1 + np.power(np.exp(1), (-1 * H)))
            w0 = w0 + alpha * (0 - G)
            w1 = w1 + alpha * (0 - G) * (x2[i])
            w2 = w2 + alpha * (0 - G) * (y2[i])
            w3 = w3 + alpha * (0 - G) * (x2[i] * y2[i])
            w4 = w4 + alpha * (0 - G) * (x2[i] ** 2)
            w5 = w5 + alpha * (0 - G) * (y2[i] ** 2)

        train_error = 0
        # Calculando error del modelo actual con ejemplos POSITIVOS
        for i in range(len(x1)):
            g, etiqueta = error(x1[i], y1[i])
            error = (g - 1) ** 2
            train_error += error
        # Calculando error del modelo actual con ejemplos NEGATIVOS
        for i in range(len(x2)):
            g, etiqueta = error(x2[i], y2[i])
            error = (g - 0) ** 2
            train_error += error
        train_error = np.sqrt(train_error)
        train_error_list.append(train_error)

        # Plotting points
        plt.plot(x1, y1, 'ro')
        plt.plot(x2, y2, 'bo')
        # Plotting model
        x = np.linspace(-10.0, 10.0, 100)
        y = np.linspace(-10.0, 10.0, 100)
        X, Y = np.meshgrid(x, y)
        F = w0 + w1 * X + w2 * Y + w3 * (X * Y) + w4 * (X ** 2) + w5 * (Y ** 2)
        # F = w0 + w1*G1(X,Y) + w2*G2(X,Y) + w3*G3(X,Y) +w4*G4(X,Y)
        plt.contour(X, Y, F, [0])
        # Plotting all
        plt.show()

    print(w0, w1, w2, w3, w4, w5)

    plt.plot(train_error_list)
