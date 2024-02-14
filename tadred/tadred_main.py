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

import argparse
import logging
import timeit

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .data_processing import create_data_norm
from .trainer import Trainer
from .utils import create_out_dirs, load_yaml, save_results_dir, set_random_seed

log = logging.getLogger(__name__)


def train_and_eval():
    """Argparser for loading .yaml config file."""
    parser = argparse.ArgumentParser(description="TADRED")
    paradd = parser.add_argument
    paradd("--cfg", type=str, default="", help="Path to YAML config file", required=True)
    args = parser.parse_args()
    args = load_yaml(args.cfg)

    run(args)


def run(args: DictConfig, pass_data: dict[str, np.ndarray] | None = None) -> dict:
    """Run TADRED with dict args and option to pass_data directly."""
    logging.basicConfig(level=logging.INFO)

    start_train_timer = timeit.default_timer()

    data, data_features_norm = create_data_norm(**args.data_norm, pass_data=pass_data)
    assert data_features_norm.n_features == args.tadred_train_eval.feature_set_sizes_Ci[0]
    log.info(OmegaConf.to_yaml(args))
    set_random_seed(seed=args.other_options.random_seed_value)

    nnet = Trainer(
        tadred_train_eval=args.tadred_train_eval,
        network=args.network,
        data_features_norm=data_features_norm,
        train_pytorch=args.train_pytorch,
        other_options=args.other_options,
    )

    results_performance = nnet.train_val_test_all(data)
    results: dict = dict()
    results.update(results_performance)
    results["args"] = args
    out_dirs = create_out_dirs(**args.output)
    save_results_dir(out_dirs["out_base_dir"], out_dirs["results_fn"], results=results)

    time_s = timeit.default_timer() - start_train_timer
    log.info(f"Total runtime (s): {time_s} (h): {time_s / 3600}")

    return results
