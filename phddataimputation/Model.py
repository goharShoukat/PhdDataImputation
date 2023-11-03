import tensorflow as tf
from tensorflow.keras import layers, utils


class ConvAndLSTMNet(tf.keras.Model):
    def __init__(self):
        super(ConvAndLSTMNet, self).__init__()

        self.twentyFourHourPrior = layers.Conv1D(
            filters=16,
            kernel_size=2,
            activation="relu",
            strides=1,
            input_shape=(24, 1),
        )

        self.twentyFourHourAfter = layers.Conv1D(
            filters=16,
            kernel_size=2,
            activation="relu",
            strides=1,
            input_shape=(24, 1),
        )

        self.concatenated = layers.Concatenate()

        self.dropout = layers.Dropout(0.2)
        self.fc1 = layers.Dense(64, activation="relu")
        self.fc2 = layers.Dense(1, activation="relu")
        self.lstm = layers.LSTM(64)
    

        self.output_layer = layers.Dense(1, activation="relu")

    def build(self, input_shape):
        input_data1 = tf.keras.Input(shape=input_shape, name="input_data1")
        input_data2 = tf.keras.Input(shape=input_shape, name="input_data2")

        x1 = self.twentyFourHourPrior(input_data1)
        x1 = layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(x1)
        x1 = self.lstm(x1)

        x2 = self.twentyFourHourAfter(input_data2)
        x2 = layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(x2)
        x2 = self.lstm(x2)

        x = self.concatenated([x1, x2])
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.output_layer(x)

        self.model = tf.keras.Model(inputs=[input_data1, input_data2], outputs=output)

    def summary(self):
        utils.plot_model(self.model, to_file='model.png')
        self.model.summary()

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, x, y, epochs, batch_size):
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)
