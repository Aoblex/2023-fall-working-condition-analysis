from sklearn.metrics import mean_squared_error, mean_absolute_error

MODEL_FOLDER = "models"
MODEL_METRICS = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
}

METRICS_FOLDER = "metrics"
PREDICTIONS_FOLDER = "predictions"
PICTURES_FOLDER = "pictures"
PARAMETERS_FOLDER = "parameters"