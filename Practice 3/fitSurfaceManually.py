import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

centers = []
sigma = 0.1
alpha = 0.0001
iterations = 1000
loop_size = 5
gauss_count = loop_size * loop_size
w_list = np.random.rand(gauss_count) - 0.5
TOTAL_NUM = 500


def error(x, y, z):
    h = calculate_h(x, y)
    return ((z - h) ** 2).mean()


def gaussian(x_val, y_val, center, sigma):
    return np.exp(-((x_val - center[0]) ** 2 / (2 * sigma ** 2) + (y_val - center[1]) ** 2 / (2 * sigma ** 2)))


def calculate_h(x, y):
    # h = w_list[0]
    h = 0
    for i in range(len(w_list)):
        h += w_list[i] * gaussian(x, y, centers[i], sigma)
    # g = 1 / (1 + np.exp(h))
    return h


if __name__ == '__main__':

    data: np.array = np.loadtxt('data_3d.csv', delimiter=',').T

    x = data[0]
    y = data[1]
    z = data[2]
    m = len(x)

    step_size = (max(x) - min(x)) / loop_size
    vertex_initial_pos = min(x) + step_size / 2
    temp = vertex_initial_pos

    for i in range(loop_size):
        for j in range(loop_size):
            centers.append([temp, vertex_initial_pos + (step_size * j)])
        temp += step_size

    mesh_x = np.linspace(min(x), max(y), 100)
    mesh_y = np.linspace(min(y), max(y), 100)
    x_surface, y_surface = np.meshgrid(mesh_x, mesh_y)

    h = calculate_h(x_surface, y_surface)

    # trace1 = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color=z))
    # trace2 = go.Surface(z=h, x=x_surface, y=y_surface, colorscale='Viridis', opacity=0.5)
    #
    # fig = go.Figure(data=[trace1, trace2])
    # fig.show()

    # Create 3D plot of the data points and the fitted curve
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color='blue', alpha=0.1)
    Z = calculate_h(x_surface, y_surface)
    ax.plot_surface(x_surface, y_surface, Z, color='red', alpha=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    # train_error_list = []
    for ite in range(iterations):
        # Positive examples
        for i in range(len(x)):
            h = calculate_h(x[i], y[i])
            for j in range(len(w_list)):
                w_list[j] = w_list[j] + alpha * ((z[i] - h) * gaussian(x[i], y[i], centers[j], sigma))

        # error_value = error(x, y, z)

        # train_error_list.append(error_value)

        # if ite % 100 == 0:
        #     print(w_list)
        #     h = calculate_h(x_surface, y_surface)
            # trace2 = go.Surface(z=h, x=x_surface, y=y_surface, colorscale='Viridis', opacity=0.5)
            # fig = go.Figure(data=[trace1, trace2])
            # fig.show()
            # Create 3D plot of the data points and the fitted curve

    h = calculate_h(x_surface, y_surface)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color='blue', alpha=0.1)
    ax.plot_surface(x_surface, y_surface, h, color='red', alpha=1)
    plt.show()

    print(w_list)
    #plt.plot(train_error_list)
