import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


class PostProcessing:
    def __init__(self, path):
        self.path = path

    def load(self):
        self.model = tf.keras.models.load_model(self.path)
        return self.model

    def normalise(self, x, y, features):
        self.scalarX = MinMaxScaler(feature_range=(0, 1))
        self.X = self.scalarX.fit_transform(x).T.reshape(-1, features, 1)
        self.scalarY = MinMaxScaler(feature_range=(0, 1))
        self.Y = self.scalarY.fit_transform(y.reshape(-1, 1))

    def predict(self, testX):
        results = self.model.predict({"input", testX})
        return results

    def denormalise(self, results):
        return self.scalarY.inverse_transform(results)
