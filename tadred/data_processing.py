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

import logging
import pickle
from enum import Enum
from pathlib import Path

import numpy as np

from .types import Data, DataFeaturesNorm

log = logging.getLogger(__name__)


def load_data(data_fil: str) -> dict:
    """Loads data dict from .pkl or .npy file."""
    data_fil_path = Path(data_fil)

    if data_fil_path.suffix == ".pkl":
        with open(data_fil_path, "rb") as f:
            data_dict = pickle.load(f)
    elif data_fil_path.suffix == ".npy":
        data_dict = np.load(data_fil_path, allow_pickle=True).item()
    else:
        raise ValueError("Data file either .pkl or .npy")
    return data_dict


def data_dict_to_array(
    data_dict: dict[str, np.ndarray], names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate and prepare data for each split"""
    names_inp = names.copy()
    data_out_inp = np.concatenate([data_dict[name] for name in names_inp], axis=0)
    data_out_inp = data_out_inp.astype(np.float32)
    log.info(f"concatenated {names_inp} on voxel dim")

    names_tar = [f"{name}_tar" for name in names]
    try:
        data_out_tar = np.concatenate([data_dict[name] for name in names_tar], axis=0)
        data_out_tar = data_out_tar.astype(np.float32)
        log.info("Target data found")
    except:
        data_out_tar = np.copy(data_out_inp)
        log.info("Target data set to input data")

    return data_out_inp, data_out_tar


def create_train_val_test(
    data_train_subjs: list[str],
    data_val_subjs: list[str],
    data_test_subjs: list[str],
    data_fil: str | None = None,
    pass_data: dict | None = None,
) -> Data:
    """Creates three splits from loading data, or from passing data."""
    if data_fil is not None:
        data_dict = load_data(data_fil)
    if pass_data is not None:
        data_dict = pass_data
    else:
        ValueError("Either provide a path to load data or pass data")

    datatrain = data_dict_to_array(data_dict, data_train_subjs)
    dataval = data_dict_to_array(data_dict, data_val_subjs)
    datatest = data_dict_to_array(data_dict, data_test_subjs)

    data = Data(
        train_x=datatrain[0],
        train_y=datatrain[1],
        val_x=dataval[0],
        val_y=dataval[1],
        test_x=datatest[0],
        test_y=datatest[1],
    )

    return data


def calc_affine_norm(
    data_np: np.ndarray,
    data_normalization: str = "original-measurement",  # TODO add enum here
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates constants for affine transformation of data.

    Args:
        data_np: Data
        data_normalization ({"original"}): Normalization from paper

    Return:
        loss_affine: Normalize data with (data - loss_affine[1])/loss_affine[0]
    """

    if data_normalization == "original-measurement":
        prctsig = 99  # Percentile for calculating normalization
        smallsig = 0  # Clamp values below this to zero
        max_val = np.float32(np.percentile(np.abs(data_np), prctsig, axis=0))
    else:
        raise ValueError("data_normalization not within available options.")

    # if smallsig > 0 rewrite so data_np[data_np<smallsig] = smallsig
    min_val = np.array(smallsig)

    loss_affine = (max_val - min_val, min_val)
    return loss_affine


def create_data_norm(
    data_train_subjs: list[str],
    data_val_subjs: list[str],
    data_test_subjs: list[str],
    data_normalization: str = "original-measurement",
    data_fil: str | None = None,
    pass_data: dict[str, np.ndarray] | None = None,
) -> tuple[Data, DataFeaturesNorm]:
    """Process data, create train-val-test-splits, other information/features of data"""

    data = create_train_val_test(
        data_train_subjs,
        data_val_subjs,
        data_test_subjs,
        data_fil,
        pass_data=pass_data,
    )

    # Other preprocessing here

    loss_affine_x = calc_affine_norm(data.train_x, data_normalization)
    loss_affine_y = calc_affine_norm(data.train_y, data_normalization)
    train_x_median = np.median(data.train_x, axis=0)

    assert loss_affine_y[1] in (0, None)

    data_features_norm = DataFeaturesNorm(
        n_features=data.train_x.shape[1],
        out_units=data.train_y.shape[1],
        loss_affine_x=loss_affine_x,
        loss_affine_y=loss_affine_y,
        train_x_median=train_x_median,
    )

    return data, data_features_norm


def tadred_data_format(
    train: np.ndarray | tuple[np.ndarray, np.ndarray],
    val: np.ndarray | tuple[np.ndarray, np.ndarray],
    test: np.ndarray | tuple[np.ndarray, np.ndarray],
):
    """Helper function, tuple-np.array or np.array for each split to TADRED format"""
    data_inp = dict(train=train, val=val, test=test)
    data = dict()
    for split in ("train", "val", "test"):
        data_split = data_inp[split]
        if isinstance(data_split, tuple) and len(data_split) == 2:
            data[split] = data_split[0]
            data[split + "_tar"] = data_split[1]
        elif isinstance(data_split, np.ndarray):
            data[split] = data_split
        else:
            raise ValueError("All splits either np.array or tuple of np.array")

    for key, val in data.items():
        assert isinstance(val, np.ndarray)
    return data
