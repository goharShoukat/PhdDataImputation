import tensorflow as tf
from tensorflow.keras import layers, models


class ConvAndLSTMNet(tf.keras.Model):
    def __init__(self, input_data1, input_data2):
        super(ConvAndLSTMNet, self).__init__()
        self.build_model(input_data1, input_data2)

    def build_model(self, input_data1, input_data2):
        twentyFourHourPrior = layers.TimeDistributed(
            layers.Conv1d(
                filter=16,
                kernel_size=2,
                activation="relu",
                stride=1,
                input_shape=(24, 1),
            )
        )
        twentyFourHourPrior = layers.MaxPooling1D(
            pool_size=3, stride=1, padding="valid"
        )(twentyFourHourPrior)

        twentyFourHourPrior = layers.Flatten()(twentyFourHourPrior)

        twentyFourHourAfter = layers.TimeDistributed(
            layers.Conv1d(
                filter=16,
                kernel_size=2,
                activation="relu",
                stride=1,
                input_shape=(24, 1),
            )
        )
        twentyFourHourAfter = layers.MaxPooling1D(
            pool_size=3, stride=1, padding="valid"
        )(twentyFourHourAfter)

        twentyFourHourAfter = layers.Flatten()(twentyFourHourAfter)

        output = layers.Concatenate()([twentyFourHourPrior, twentyFourHourAfter])
        output = layers.Dense(64, activation="relu")(output)
        output = layers.Dense(1, activation="relu")(output)

        output = layers.LSTM(64)(output)
        output = layers.Dense(1, activation='relu')(output)

        self.model = tf.keras.Model(inputs=[input_data1, input_data2], outputs=output)
