from random import random
from random import seed
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def error(x_error, y_error):
    # h_error = w0 + w1 * x_error + w2 * x_error + w3 * (x_error * y_error) + w4 * (x_error ** 2) + w5 * (x_error ** 2)
    h_error = (
                    w0 +
                    w1 * gaussian(x_error, y_error, gauss_centers[0], sigma) +
                    w2 * gaussian(x_error, y_error, gauss_centers[1], sigma) +
                    w3 * gaussian(x_error, y_error, gauss_centers[2], sigma) +
                    w4 * gaussian(x_error, y_error, gauss_centers[3], sigma) +
                    w5 * gaussian(x_error, y_error, gauss_centers[4], sigma) +
                    w6 * gaussian(x_error, y_error, gauss_centers[5], sigma) +
                    w7 * gaussian(x_error, y_error, gauss_centers[6], sigma) +
                    w8 * gaussian(x_error, y_error, gauss_centers[7], sigma) +
                    w8 * gaussian(x_error, y_error, gauss_centers[8], sigma)
                 )
    g_error = 1 / (1 + np.power(np.exp(1), (-1 * h_error)))
    if g_error >= 0.5:
        return g_error, "Red"
    else:
        return g_error, "Blue"


def gaussian(x_val, y_val, center, sigma_error):
    return np.exp(-((x_val - center[0]) ** 2 / (2 * sigma_error ** 2) + (y_val - center[1]) ** 2 / (2 * sigma_error ** 2)))


if __name__ == '__main__':
    # seed random number generator
    seed(1)

    X = []
    Y = []

    x1 = []
    y1 = []

    x2 = []
    y2 = []

    TOTAL_NUM = 500

    for _ in range(TOTAL_NUM):
        X.append(random() * 20 - 10)
        Y.append(random() * 20 - 10)

    for i in range(TOTAL_NUM):
        dist = np.sqrt(X[i] * X[i] + Y[i] * Y[i])
        if dist < 4:
            x1.append(X[i])
            y1.append(Y[i])
        elif dist > 6:
            x2.append(X[i])
            y2.append(Y[i])

    alpha = 0.0009
    iterations = 100
    sigma = 1

    w0 = 0.8
    w1 = random() - 0.5
    w2 = random() - 0.5
    w3 = random() - 0.5
    w4 = random() - 0.5
    w5 = random() - 0.5
    w6 = random() - 0.5
    w7 = random() - 0.5
    w8 = random() - 0.5
    w9 = random() - 0.5

    x = np.linspace(-10.0, 10.0, 100)
    y = np.linspace(-10.0, 10.0, 100)
    X, Y = np.meshgrid(x, y)

    temp_miu_x = np.linspace(-10.0, 10.0, 9)
    temp_miu_y = np.linspace(-10.0, 10.0, 9)

    gauss_centers = []

    for i in range(len(temp_miu_x)):
        gauss_centers.append([temp_miu_x[i], temp_miu_y[i]])

    G1 = gaussian(X, Y, gauss_centers[0], sigma)
    G2 = gaussian(X, Y, gauss_centers[1], sigma)
    G3 = gaussian(X, Y, gauss_centers[2], sigma)
    G4 = gaussian(X, Y, gauss_centers[3], sigma)
    G5 = gaussian(X, Y, gauss_centers[4], sigma)
    G6 = gaussian(X, Y, gauss_centers[5], sigma)
    G7 = gaussian(X, Y, gauss_centers[6], sigma)
    G8 = gaussian(X, Y, gauss_centers[7], sigma)
    G9 = gaussian(X, Y, gauss_centers[8], sigma)

    F = w0 + w1 * G1 + w2 * G2 + w3 * G3 + w4 * G4 + w4 * G5 + w4 * G6 + w4 * G7 + w4 * G8 + w4 * G9
    G = 1 / (1 + np.exp(-F))

    z1 = np.full(len(x1), 0.9)
    z2 = np.full(len(x2), 1)
    z = np.concatenate((z1, z2))

    trace1 = go.Scatter3d(x=x1 + x2, y=y1 + y2, z=z, mode='markers', marker=dict(size=5, color=z))
    trace2 = go.Surface(z=G, x=X, y=Y, colorscale='Viridis', showscale=False, opacity=0.5)

    fig = go.Figure(data=[trace1, trace2])
    fig.show()

    train_error_list = []
    for ite in range(iterations):
        # Positive examples
        for i in range(len(x1)):
            H = (
                    w0 +
                    w1 * gaussian(x1[i], y1[i], gauss_centers[0], sigma) +
                    w2 * gaussian(x1[i], y1[i], gauss_centers[1], sigma) +
                    w3 * gaussian(x1[i], y1[i], gauss_centers[2], sigma) +
                    w4 * gaussian(x1[i], y1[i], gauss_centers[3], sigma) +
                    w5 * gaussian(x1[i], y1[i], gauss_centers[4], sigma) +
                    w6 * gaussian(x1[i], y1[i], gauss_centers[5], sigma) +
                    w7 * gaussian(x1[i], y1[i], gauss_centers[6], sigma) +
                    w8 * gaussian(x1[i], y1[i], gauss_centers[7], sigma) +
                    w8 * gaussian(x1[i], y1[i], gauss_centers[8], sigma)
                 )
            G = 1 / (1 + np.exp(-H))
            #w0 = w0 + alpha * (1 - G)
            w1 = w1 + alpha * (1 - G) * gaussian(x1[i], y1[i], gauss_centers[0], sigma)
            w2 = w2 + alpha * (1 - G) * gaussian(x1[i], y1[i], gauss_centers[1], sigma)
            w3 = w3 + alpha * (1 - G) * gaussian(x1[i], y1[i], gauss_centers[2], sigma)
            w4 = w4 + alpha * (1 - G) * gaussian(x1[i], y1[i], gauss_centers[3], sigma)
            w5 = w5 + alpha * (1 - G) * gaussian(x1[i], y1[i], gauss_centers[4], sigma)
            w6 = w6 + alpha * (1 - G) * gaussian(x1[i], y1[i], gauss_centers[5], sigma)
            w7 = w7 + alpha * (1 - G) * gaussian(x1[i], y1[i], gauss_centers[6], sigma)
            w8 = w8 + alpha * (1 - G) * gaussian(x1[i], y1[i], gauss_centers[7], sigma)
            w9 = w9 + alpha * (1 - G) * gaussian(x1[i], y1[i], gauss_centers[8], sigma)
        # Negative examples
        for i in range(len(x2)):
            H = (
                    w0 +
                    w1 * gaussian(x2[i], y2[i], gauss_centers[0], sigma) +
                    w2 * gaussian(x2[i], y2[i], gauss_centers[1], sigma) +
                    w3 * gaussian(x2[i], y2[i], gauss_centers[2], sigma) +
                    w4 * gaussian(x2[i], y2[i], gauss_centers[3], sigma) +
                    w5 * gaussian(x2[i], y2[i], gauss_centers[4], sigma) +
                    w6 * gaussian(x2[i], y2[i], gauss_centers[5], sigma) +
                    w7 * gaussian(x2[i], y2[i], gauss_centers[6], sigma) +
                    w8 * gaussian(x2[i], y2[i], gauss_centers[7], sigma) +
                    w8 * gaussian(x2[i], y2[i], gauss_centers[8], sigma)
            )
            G = 1 / (1 + np.exp(-H))
            # w0 = w0 + alpha * (1 - G)
            w1 = w1 + alpha * (1 - G) * gaussian(x2[i], y2[i], gauss_centers[0], sigma)
            w2 = w2 + alpha * (1 - G) * gaussian(x2[i], y2[i], gauss_centers[1], sigma)
            w3 = w3 + alpha * (1 - G) * gaussian(x2[i], y2[i], gauss_centers[2], sigma)
            w4 = w4 + alpha * (1 - G) * gaussian(x2[i], y2[i], gauss_centers[3], sigma)
            w5 = w5 + alpha * (1 - G) * gaussian(x2[i], y2[i], gauss_centers[4], sigma)
            w6 = w6 + alpha * (1 - G) * gaussian(x2[i], y2[i], gauss_centers[5], sigma)
            w7 = w7 + alpha * (1 - G) * gaussian(x2[i], y2[i], gauss_centers[6], sigma)
            w8 = w8 + alpha * (1 - G) * gaussian(x2[i], y2[i], gauss_centers[7], sigma)
            w9 = w9 + alpha * (1 - G) * gaussian(x2[i], y2[i], gauss_centers[8], sigma)

        train_error = 0

        # Calculating error of the current model with POSITIVE examples
        for i in range(len(x1)):
            g, tag = error(x1[i], y1[i])
            error_value = (g - 1) ** 2
            train_error += error_value

        # Calculating error of the current model with NEGATIVE examples
        for i in range(len(x2)):
            g, tag = error(x2[i], y2[i])
            error_value = (g - 0) ** 2
            train_error += error_value
        train_error = np.sqrt(train_error)
        train_error_list.append(train_error)

        if ite % 10 == 0:
            F = w0 + w1 * G1 + w2 * G2 + w3 * G3 + w4 * G4 + w4 * G5 + w4 * G6 + w4 * G7 + w4 * G8 + w4 * G9
            G = 1 / (1 + np.exp(-F))

            trace2 = go.Surface(z=G, x=X, y=Y, colorscale='Viridis', showscale=False, opacity=0.5)
            fig = go.Figure(data=[trace1, trace2])
            fig.show()

    print(w0, w1, w2, w3, w4)
    plt.plot(train_error_list)
