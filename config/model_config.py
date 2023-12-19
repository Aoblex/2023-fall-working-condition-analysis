from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

MODEL_FOLDER = "models"
MODEL_METRICS = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "r2": r2_score,
    "evs": explained_variance_score,
}

METRICS_FOLDER = "metrics"
PREDICTIONS_FOLDER = "predictions"
PICTURES_FOLDER = "pictures"
PARAMETERS_FOLDER = "parameters"