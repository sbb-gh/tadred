"""
Copyright 2024 Stefano B. Blumberg

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from dataclasses import dataclass, field

import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DataNorm:
    data_fil: str | None = None  # Path to data dictionary
    data_train_subjs: tuple[str] = ("train",)  # Data elements for training
    data_val_subjs: tuple[str] = ("val",)  # Data elements for validation
    data_test_subjs: tuple[str] = ("test",)  # Data elements for testing
    data_normalization: str = "original-measurement"


@dataclass
class Output:
    out_base: str | None = None  # Outputs saved directory, set to None to not save results file
    proj_name: str = "tst"  # Output proj_name/run_name
    run_name: str = "def"  # Output proj_name/run_name


@dataclass
class NetworkArgs:
    num_units_score: list[int] = MISSING  # Middle units in Score Network S, [-1] to switch off
    num_units_task: list[int] = MISSING  # Middle units in Task Network T, set to [-1] to switch off
    score_activation: str = "doublesigmoid"  # Activation function for score $ \sigma $ in paper


@dataclass
class TadredTrainEval:
    feature_set_sizes_Ci: list[int] = MISSING  # Values of feature subsets considered C_1, C_2,...
    feature_set_sizes_evaluated: list[int] = MISSING  # Evaluate at this feature subset C
    epochs: int = 1000  # Total training epochs E in paper
    epochs_fix_sigma: int = 25  # Fix score after epoch, E_1 in paper
    epochs_decay_sigma: int = (
        10  # Progressively set score to be sample independent across number epochs, E_2 - E_1 in paper
    )
    epochs_decay: int = 10  # Progressively modify mask across number epochs, E_3 - E_2 in paper


@dataclass
class DataloaderParams:
    batch_size: int = 1500
    num_workers: int = 0


@dataclass
class OptimizerParams:
    lr: float = 0.0001


@dataclass
class TrainPytorch:
    dataloader_params: DataloaderParams = field(default_factory=DataloaderParams)
    optimizer_params: OptimizerParams = field(default_factory=OptimizerParams)


@dataclass
class OtherOptions:
    random_seed_value: int = 0
    no_gpu: bool = False  # Run on CPU
    save_output: bool = False  # Saves prediction on test data - may fill up disk space


@dataclass
class MyConfig:
    data_norm: DataNorm = field(default_factory=DataNorm)
    output: Output = field(default_factory=Output)
    network: NetworkArgs = field(default_factory=NetworkArgs)
    tadred_train_eval: TadredTrainEval = field(default_factory=TadredTrainEval)
    train_pytorch: TrainPytorch = field(default_factory=TrainPytorch)
    other_options: OtherOptions = field(default_factory=OtherOptions)


@dataclass
class Data:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


@dataclass
class DataFeaturesNorm:
    n_features: int
    out_units: int
    loss_affine_x: tuple[np.ndarray, np.ndarray]
    loss_affine_y: tuple[np.ndarray, np.ndarray]
    train_x_median: np.ndarray


cs = ConfigStore.instance()
cs.store(name="cfg", node=MyConfig)
