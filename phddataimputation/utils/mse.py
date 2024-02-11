from sklearn.metrics import mean_squared_error


def mse(df, true_column_name, predicted_column_name):
    return mean_squared_error(df[true_column_name], df[predicted_column_name])
