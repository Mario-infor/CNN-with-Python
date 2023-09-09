import tensorflow as tf
from keras import layers
from keras.utils import to_categorical
from sklearn import datasets
from tensorflow import keras
import plotly.express as px


class WeightHistoryCallback(keras.callbacks.Callback):
    def __init__(self, layer_name):
        super(WeightHistoryCallback, self).__init__()
        self.layer_name = layer_name
        self.weight_history = []

    def on_epoch_end(self, epoch, logs=None):
        # Obtener el valor del peso específico al final de cada época
        model_weights = self.model.get_layer(self.layer_name).get_weights()
        self.weight_history.append(model_weights)


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    target = iris.target
    Y = to_categorical(target, dtype="uint8")

    model = keras.Sequential(
        [
            layers.Dense(3, activation="sigmoid", name="layer1", input_shape=(4,)),
            # layers.Dense(3, activation="sigmoid", name="layer2"),
            # layers.Dense(3, activation="sigmoid", name="layer3"),
            # layers.Dense(3, activation="sigmoid", name="layer4"),
            layers.Dense(3, activation="softmax", name="output_layer"),
        ]
    )
    model.summary()

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.03)
    loss = tf.keras.losses.MeanSquaredError()

    custom_callback = WeightHistoryCallback(layer_name='layer1')

    model.compile(optimizer, loss)
    history = model.fit(X, Y, epochs=500, callbacks=[custom_callback])

    neuron_weights_in_history = []
    single_weight_history = []

    i = 0
    j = 0
    k = 0
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

    fig = px.line(y=neuron_weights_in_history[0], title='Gráfico de Dispersión 2D')
    fig.show()
