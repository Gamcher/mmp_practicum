from pathlib import Path

from pydantic import BaseModel
from typing import Optional, Union, Literal


class ExistingExperimentsResponse(BaseModel):
    """
    Response model for existing experiments.

    Attributes:
        location (Path): The directory path where the experiments are stored.
        experiment_names (list[str]): A list of names of the existing experiments. Defaults to an empty list.
        abs_paths (list[Path]): A list of absolute paths to the experiment directories. Defaults to an empty list.
    """

    location: Path
    experiment_names: list[str] = []
    abs_paths: list[Path] = []


class ExperimentConfig(BaseModel):
    """
    Config for experiments.

    Attributes:
        name (str): The name of experiment.
        ml_model (str): The name of model which will used in experiment.
        n_estimators (int): The count of trees in models. Defaults to 100.
        max_depth (int): The maximum of depth for each tree in model. Defaults to 3.
        max_features (int/"log2"/"sqrt"): The number of features to be selected at each step for each tree in the model.
        learning_rate (float): The learning rate of model. Defaults to 0.1.
        target_column (str): The name of target columns in data.
    """

    name: str
    ml_model: Literal["Gradient Boosting", "Random Forest"]
    n_estimators: Optional[int] = 100
    max_depth: Optional[int] = 3
    max_features: Union[int, str]
    learning_rate: Optional[float] = 0.1
    target_column: str


class ConvergenceHistoryResponse(BaseModel):
    """
    The convergence history of experiment.

    Attributes:
        val (list[float]): A list of RMLSE on validation data for each tree.
        train (list[float]): A list of RMLSE on train data for each tree.
        time (list[float]): A list of training time for each tree.
    """

    val: Optional[list[float]] = None
    train: list[float]
    time: list[float]
