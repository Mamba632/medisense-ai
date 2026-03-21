import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import keras

from src.config import DL_PARAMS


def build_dl_model(input_dim: int, dropout_rate: float = DL_PARAMS["dropout_rate"]):
    inputs = keras.Input(shape=(input_dim,))
    x = inputs

    for units in DL_PARAMS["hidden_layers"]:
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    output = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=DL_PARAMS["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self, input_dim: int = 10, dropout_rate: float = DL_PARAMS["dropout_rate"]
    ):
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.model_ = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.input_dim = X.shape[1]
        self.model_ = build_dl_model(self.input_dim, self.dropout_rate)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=DL_PARAMS["patience"],
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5
            ),
        ]

        self.model_.fit(
            X,
            y,
            epochs=DL_PARAMS["max_epochs"],
            batch_size=DL_PARAMS["batch_size"],
            validation_split=0.15,
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict_proba(self, X):
        prob_positive = self.model_.predict(X, verbose=0).ravel()
        return np.column_stack([1 - prob_positive, prob_positive])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
