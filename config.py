from typing import List, Optional

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Config(object):
    """
    Attributes
    ----------

    decay_rate:
        Decay rate of the momentum

    data_paths:
        List of paths to input files. The input files can be split into
        multiple csv as long as they have same header and format.
        When loading the data through `get_train_test_datasets` one can
        choose whether the csv have to be merged (when loaded) and then
        transformed into windows or merged only after windowing. This
        was done since the SML2010 dataset is split into two csvs with the
        second starting at a timestep far later than the first one

    target_cols:
        The name of the columns that are the target values (y) for the
        prediction. When the dataset is returned from
        `get_train_test_datasets` the X will contain values of the columns in
        (header minus (target_cols union drop_cols)) while the y values of the
        `target_cols`.

    drop_cols:
        The name of the columns to be dropped from the csv.
        In the case of SML2010 some columns are 0, such as the ones regarding
        Exterior_Entalpic_*

    n:
        Number of driving series

    m:
        The size of the hidden state of the encoder

    p:
        The size of the hidden state of the decoder

    sep:
        The pattern separating columns in the csvs

    T:
        The number of past values the predictor can perceive. T values
        of the driving series (X) and T-1 values for the target series.
        The T-th values of the target series (y) are the ones to be predicted

    learning_rate:
        Learning rate for optimizing the parameters

    decay_frequency:

    max_gradient_norm:
        Used in gradient clipping

    optimizer:
        Optimizer name

    batch_size:
        Number of windows to be processed at the same time

    num_epochs:
        Number of epochs to train the network

    log_dir:
        Directory for logging

    train_ratio:
        Portion of the data to be used as training set. The remainder of
        the portion is equally split into test and validation.

    report_frequency:
        Print loss and train speed each [this param] batches

    plot_frequency:
        Plot true and predicted curves each [this param] epochs
    """

    decay_rate: float
    data_paths: List[str]
    target_cols: List[str]
    drop_cols: Optional[List[str]] = field(default_factory=list)
    n: Optional[
        int
    ] = 0  # Set at runtime by `data_loader.get_train_test_dataset`
    m: int = 64
    p: int = 64
    sep: str = ","
    T: int = 10
    learning_rate: float = 0.001
    decay_frequency: int = 1000
    max_gradient_norm: float = 5
    optimizer: str = "adam"
    batch_size: int = 128
    num_epochs: int = 10
    log_dir: str = "log/"
    train_ratio: float = 0.8
    report_frequency: int = 50
    plot_frequency: int = 10
    seed: int = 42

    @property
    def usecols(self):
        path = self.data_paths[0]
        with open(path) as f:
            header = f.readline().strip().split(self.sep)
        return [col for col in header if col not in self.drop_cols]

    @property
    def driving_series(self):
        return [col for col in self.usecols if col not in self.target_cols]


# Test
if __name__ == "__main__":
    with open("conf/experiment2.json") as f:
        c = Config.from_json(f.read())
    print(c.to_json())
