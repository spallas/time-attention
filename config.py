from typing import List, Optional

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Config(object):
    """
    Attributes
    ----------
    # TODO(xhebraj) n shouldn't be in the config. It should be inferred
    n:
        Number of driving series

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

    batch_size:
        Number of windows to be processed at the same time

    num_epochs:
        Number of epochs to train the network

    log_dir:
        Directory for logging

    """

    n: Optional[int]  # Set at runtime
    decay_rate: float
    data_paths: List[str]
    target_cols: List[str]
    drop_cols: Optional[List[str]] = field(default_factory=list)
    m: int = 64  # TODO(xhebraj) set optimal values of static attributes
    p: int = 16
    sep: str = ","
    T: int = 10
    learning_rate: float = 0.01
    decay_frequency: int = 100
    batch_size: int = 4
    num_epochs: int = 10
    log_dir: str = "log/"


# Test
if __name__ == "__main__":
    with open("conf/experiment2.json") as f:
        c = Config.from_json(f.read())
    print(c.to_json())
