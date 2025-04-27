import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.tree import DecisionTreeRegressor

from .utils import ConvergenceHistory, rmsle, whether_to_stop


class RandomForestMSE:
    def __init__(
        self, n_estimators: int, random_state: int = 42, tree_params: dict[str, Any] | None = None
    ) -> None:
        """
        Handmade random forest regressor.

        Classic ML algorithm that trains a set of independent tall decision trees and averages its predictions. Employs scikit-learn `DecisionTreeRegressor` under the hood.

        Args:
            n_estimators (int): Number of trees in the forest.
            tree_params (dict[str, Any] | None, optional): Parameters for sklearn trees. Defaults to None.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state

        if tree_params is None:
            tree_params = {}

        if 'max_features' not in tree_params:
            tree_params['max_features'] = 'sqrt'

        tree_params['random_state'] = random_state

        self.forest = [
            DecisionTreeRegressor(**tree_params) for _ in range(n_estimators)
        ]

        self._fitted_trees = 0

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64] | None = None,
        y_val: npt.NDArray[np.float64] | None = None,
        trace: bool | None = None,
        patience: int | None = None,
    ) -> ConvergenceHistory | None:
        """
        Train an ensemble of trees on the provided data.

        Args:
            X (npt.NDArray[np.float64]): Objects features matrix, array of shape (n_objects, n_features).
            y (npt.NDArray[np.float64]): Regression labels, array of shape (n_objects,).
            X_val (npt.NDArray[np.float64] | None, optional): Validation set of objects, array of shape (n_val_objects, n_features). Defaults to None.
            y_val (npt.NDArray[np.float64] | None, optional): Validation set of labels, array of shape (n_val_objects,). Defaults to None.
            trace (bool | None, optional): Whether to calculate rmsle while training. True by default if validation data is provided. Defaults to None.
            patience (int | None, optional): Number of training steps without decreasing the train loss (or validation if provided), after which to stop training. Defaults to None.

        Returns:
            ConvergenceHistory | None: Instance of `ConvergenceHistory` if `trace=True` or if validation data is provided.
        """
        np.random.seed(self.random_state)

        history = ConvergenceHistory()
        history['train'] = []
        history['time'] = []

        if X_val is not None:
            trace = True
            history['val'] = []

        start_time = time.time()

        y_pred_t = np.zeros(shape=X.shape[0])

        if (X_val is not None) and (y_val is not None):
            y_pred_v = np.zeros(shape=X_val.shape[0])

        for t in range(self.n_estimators):
            index = np.random.randint(0, X.shape[0], size=X.shape[0])

            self.forest[t].fit(X[index, :], y[index])
            self._fitted_trees += 1

            y_pred_t += self.forest[t].predict(X)

            history["train"].append(rmsle(y, y_pred_t / (self._fitted_trees)))
            history['time'].append(time.time() - start_time)

            if (X_val is not None) and (y_val is not None):
                y_pred_v += self.forest[t].predict(X_val)
                history["val"].append(
                    rmsle(y_val, y_pred_v / (self._fitted_trees)))

            if patience:
                if whether_to_stop(convergence_history=history,
                                   patience=patience):
                    break

        if trace:
            return history

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Make prediction with ensemble of trees.

        All the trees make their own predictions which then are averaged.

        Args:
            X (npt.NDArray[np.float64]): Objects' features matrix, array of shape (n_objects, n_features).

        Returns:
            npt.NDArray[np.float64]: Predicted values, array of shape (n_objects,).
        """
        y_pred = np.zeros(shape=X.shape[0])

        for t in range(self._fitted_trees):
            y_pred += self.forest[t].predict(X)

        return y_pred / self._fitted_trees

    def dump(self, dirpath: str) -> None:
        """
        Save the trained model to the specified directory.

        Args:
            dirpath (str): Path to the directory where the model will be saved.
        """
        path = Path(dirpath)
        path.mkdir(parents=True)

        params = {"n_estimators": self.n_estimators}
        with (path / "params.json").open("w") as file:
            json.dump(params, file, indent=4)

        trees_path = path / "trees"
        trees_path.mkdir()
        for i, tree in enumerate(self.forest):
            joblib.dump(tree, trees_path / f"tree_{i:04d}.joblib")

    @classmethod
    def load(cls, dirpath: str) -> "RandomForestMSE":
        """
        Load a trained model from the specified directory.

        Args:
            dirpath (str): Path to the directory where the model is saved.

        Returns:
            RandomForestMSE: An instance of the loaded model.
        """
        with (Path(dirpath) / "params.json").open() as file:
            params = json.load(file)
        instance = cls(params["n_estimators"])

        trees_path = Path(dirpath) / "trees"

        instance.forest = [
            joblib.load(trees_path / f"tree_{i:04d}.joblib")
            for i in range(params["n_estimators"])
        ]

        return instance
