import tensorflow as tf
from keras import layers
from keras.utils import to_categorical
from sklearn import datasets
from tensorflow import keras
import plotly.express as px


class WeightHistoryCallback(keras.callbacks.Callback):
    # Constructor for the class
    def __init__(self, layer_name):
        super(WeightHistoryCallback, self).__init__()
        self.layer_name = layer_name
        self.weight_history = []  # Array to save the weight of the model on each epoch

    def on_epoch_end(self, epoch, logs=None):
        # Get specific weight value at the end of each epoch
        model_weights = self.model.get_layer(self.layer_name).get_weights()
        self.weight_history.append(model_weights)


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    target = iris.target
    Y = to_categorical(target, dtype="uint8")

    # Building the model of the neural network using Keras
    model_1 = keras.Sequential(
        [
            layers.Dense(3, activation="sigmoid", name="layer1", input_shape=(4,)),
            # layers.Dense(3, activation="sigmoid", name="layer2"),
            # layers.Dense(3, activation="sigmoid", name="layer3"),
            # layers.Dense(3, activation="sigmoid", name="layer4"),
            layers.Dense(3, activation="softmax", name="output_layer"),
        ]
    )
    model_1.summary()

    # Building the model of the neural network using Keras
    model_2 = keras.Sequential(
        [
            layers.Dense(3, activation="relu", name="layer1", input_shape=(4,)),
            # layers.Dense(3, activation="relu", name="layer2"),
            # layers.Dense(3, activation="relu", name="layer3"),
            # layers.Dense(3, activation="relu", name="layer4"),
            layers.Dense(3, activation="softmax", name="output_layer"),
        ]
    )
    model_2.summary()

    # Building the model of the neural network using Keras
    model_3 = keras.Sequential(
        [
            layers.Dense(3, activation="tanh", name="layer1", input_shape=(4,)),
            # layers.Dense(3, activation="tanh", name="layer2"),
            # layers.Dense(3, activation="tanh", name="layer3"),
            # layers.Dense(3, activation="tanh", name="layer4"),
            layers.Dense(3, activation="softmax", name="output_layer"),
        ]
    )
    model_3.summary()

    # Optimizer Stochastic Gradient Decent
    optimizer_1 = tf.keras.optimizers.SGD(learning_rate=0.03)
    optimizer_2 = tf.keras.optimizers.Adam(learning_rate=0.03)
    loss = tf.keras.losses.MeanSquaredError()

    # Preparing the callback for retrieving the weights of the model on each epoch
    custom_callback = WeightHistoryCallback(layer_name='layer1')

    model_2.compile(optimizer_2, loss)
    history = model_2.fit(X, Y, epochs=500, callbacks=[custom_callback])

    # Array of weights through all epochs, each row responds to one weight´s history
    neuron_weights_in_history = []
    single_weight_history = []

    i = 0
    j = 0
    k = 0

    # Loop for ordering the weights values in order to plot them afterward
    while i < len(custom_callback.weight_history):
        j = 0
        while j < len(custom_callback.weight_history[i][0]):
            while k < len(custom_callback.weight_history[i][0][j]):
                single_weight_history.append(custom_callback.weight_history[i][0][j][k])
                i += 1
                if i == len(custom_callback.weight_history):
                    i = 0
                    break
            neuron_weights_in_history.append(single_weight_history.copy())
            single_weight_history.clear()
            j += 1
        k += 1
        if k == len(custom_callback.weight_history[i][0][0]):
            break

    # Using Plotly to plot the weights´s history
    fig = px.line(y=neuron_weights_in_history, title='Gráfico de Dispersión 2D')
    fig.show()
