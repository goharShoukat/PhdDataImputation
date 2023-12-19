import tensorflow as tf
from tensorflow.keras import layers, utils


class LSTMNet(tf.keras.Model):
    def __init__(self, neurons):
        super(LSTMNet, self).__init__()
        self.lstm = layers.LSTM(neurons)
        self.fc = layers.Dense(1, activation="relu")
        self.dropout = layers.Dropout(0.2)
        self.output_layer = layers.Dense(1, activation="linear")

    def build(self, input_shape):
        input_data = tf.keras.Input(shape=input_shape, name="input")
        x = self.lstm(input_data)
        x = self.fc(x)

        output = self.output_layer(x)

        self.model = tf.keras.Model(inputs=input_data, outputs=output)

    def summary(self, path):
        utils.plot_model(self.model, to_file="{}/architecture.png".format(path))
        self.model.summary()

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, x, y, epochs, batch_size):
        self.model.fit(
            {
                "input": x,
            },
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
        )

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, path, format):
        self.model.save(path, save_format=format)
