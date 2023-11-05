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

        self.dropout1 = layers.Dropout(0.2)
        self.dropout2 = layers.Dropout(0.2)
        self.fc1 = layers.Dense(2, activation="relu")
        self.fc2 = layers.Dense(4, activation="relu")
        self.fc3 = layers.Dense(16, activation="relu")
        self.fc4 = layers.Dense(64, activation="relu")
        self.fc5 = layers.Dense(128, activation="relu")
        self.fc6 = layers.Dense(64, activation="relu")
        self.lstmPrior = layers.LSTM(64, return_sequences=True)
        self.lstmAfter = layers.LSTM(64, return_sequences=True)
        self.fc7 = layers.Dense(2, activation="relu")

        self.output_layer = layers.Dense(1, activation="relu")

    def build(self, input_shape):
        input_data1 = tf.keras.Input(shape=input_shape, name="input_data1")
        input_data2 = tf.keras.Input(shape=input_shape, name="input_data2")

        x1 = self.twentyFourHourPrior(input_data1)
        x1 = layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(x1)
        x1 = self.lstmPrior(x1)

        x2 = self.twentyFourHourAfter(input_data2)
        x2 = layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(x2)
        x2 = self.lstmAfter(x2)

        x = self.concatenated([x1, x2])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dropout1(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.dropout2(x)
        x = self.fc7(x)

        output = self.output_layer(x)

        self.model = tf.keras.Model(inputs=[input_data1, input_data2], outputs=output)

    def summary(self, path):
        utils.plot_model(self.model, to_file="{}/architecture.png".format(path))
        self.model.summary()

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, x1, x2, y, epochs, batch_size):
        self.model.fit(
            {
                "input_data1": x1,
                "input_data2": x2,
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
