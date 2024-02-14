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

import copy
import logging
import timeit

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .networks import TADREDNet
from .types import Data, DataFeaturesNorm, NetworkArgs, OtherOptions, TadredTrainEval, TrainPytorch

log = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    """Simple input-target Pytorch dataset"""

    def __init__(self, data_x: np.ndarray, data_y: np.ndarray | None = None):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        if self.data_y is not None:
            return self.data_x[index, :], self.data_y[index, :]
        else:
            return self.data_x[index, :], np.inf

    def __len__(self):
        return self.data_x.shape[0]


class Trainer:
    """TADRED training class"""

    def __init__(
        self,
        tadred_train_eval: TadredTrainEval,
        network: NetworkArgs,
        data_features_norm: DataFeaturesNorm,
        train_pytorch: TrainPytorch,
        other_options: OtherOptions,
    ):
        """
        Args:
        tadred_train_eval, network, train_pytorch, other_options: Top-level arguments in cfg file.
        data_features_norm: Information about the data passed to networks.
        """
        self.tadred_train_eval = tadred_train_eval
        self.network = network
        self.data_features_norm = data_features_norm
        self.train_pytorch = train_pytorch
        self.other_options = other_options

    def _create_model(self):
        model = TADREDNet(
            **self.network,
            **self.data_features_norm.__dict__,
            tadred_train_eval=self.tadred_train_eval,
        )
        log.info(model)
        self.model = model.to(self.device)

    def _create_optimizer(self):
        # Only have optimizer for now
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), **self.train_pytorch.optimizer_params
        )

    def _create_dataloaders(
        self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray
    ):
        """Create PyTorch dataloaders for training and validation"""
        train_data = Dataset(train_x, train_y)
        val_data = Dataset(val_x, val_y)
        dataloader_params: dict = OmegaConf.to_container(self.train_pytorch.dataloader_params)
        self.train_loader = DataLoader(train_data, **dataloader_params)
        self.val_loader = DataLoader(val_data, **dataloader_params)

    def _train_val_epoch(self, epoch: int) -> tuple[float, float]:
        """Train and validate a single TADRED epoch"""
        train_losses = []
        self.model.train()
        self.model.on_epoch_begin()
        start_epoch = timeit.default_timer()
        # Forwards and backwards pass for training epoch
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.forward_and_backward(x, y)
            self.optimizer.step()
            train_losses.append(float(loss))
            self.model.on_batch_end_train()
        # Forwards and backwards pass for validation epoch
        with torch.no_grad():
            val_losses = []
            self.model.eval()
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.model.forward_and_loss(x, y)
                val_losses.append(float(loss))
        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)

        self.model.on_epoch_end()
        print(
            f"Epoch:{epoch:.0f} train_loss:{mean_train:.3f} val_loss:{mean_val:.3f} "
            f"time:{timeit.default_timer() - start_epoch:.3f}"
        )
        return float(mean_train), float(mean_val)

    def _early_stopping(self, epoch, epoch_val):
        """Early stopping on validation performance for training"""
        patience = 20
        start_es = (
            self.tadred_train_eval.epochs_fix_sigma
            + self.tadred_train_eval.epochs_decay_sigma
            + self.tadred_train_eval.epochs_decay
            + 1
        )
        if epoch >= start_es:
            if epoch_val[-1] == min(epoch_val[start_es:]):
                log.info("Cached best_state_dict")
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
        if len(epoch_val) > (start_es + patience):
            if epoch_val[-patience - 1] < min(epoch_val[-patience:]):
                return True
        return False

    def _train_val_step(self):
        """Perform TADRED step t."""
        epoch_val = []
        timer_t = timeit.default_timer()
        self.model.on_step_begin()
        self.best_state_dict = None
        for epoch in range(self.tadred_train_eval.epochs):
            _, mean_val = self._train_val_epoch(epoch=epoch)
            epoch_val.append(mean_val)
            if self._early_stopping(epoch, epoch_val):
                break

        self.model.load_state_dict(self.best_state_dict)
        self.model.on_step_end()
        log.info(
            "\n"
            f"Finished training step {self.model.t:.0f} \n"
            f"m = {float(torch.sum(self.model.get('m') == 1)):.0f} \n"
            f"step time:{timeit.default_timer() - timer_t:.3f}"
        )

    def _eval_step(self, data_x: np.ndarray, data_y: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluation on test data."""
        m = self.model.get("m")
        log.info(f"m {len(torch.where(m==1)[0])} ones {len(torch.where(m==0)[0])} zeros")
        data_x = data_x * m.cpu().numpy()  # just to make sure

        self.model.eval()
        with torch.no_grad():
            out = []
            loader = torch.utils.data.DataLoader(
                Dataset(data_x),
                batch_size=self.train_pytorch.dataloader_params.batch_size,
                shuffle=False,
            )
            for x, _ in loader:
                x = x.to(self.device)

                y_pred = self.model.forward_eval(x, score=1)
                out.append(y_pred.cpu())
            pred_all = torch.cat(out).detach()
            loss = self.model.loss_fct(torch.tensor(data_y), pred_all)
        return float(loss), pred_all.numpy()

    def train_val_test_all(self, data: Data) -> dict[int, dict]:
        """Complete optimization for TADRED"""
        if self.other_options.no_gpu:
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Run training on: {self.device}")

        self._create_model()
        self._create_optimizer()
        self._create_dataloaders(data.train_x, data.train_y, data.val_x, data.val_y)

        results = {}

        # Step t=1,...,T feature set size Ci
        for t, Ci in enumerate(self.tadred_train_eval.feature_set_sizes_Ci, 1):
            self._train_val_step()

            if Ci in self.tadred_train_eval.feature_set_sizes_evaluated:
                val_joint, _ = self._eval_step(data.val_x, data.val_y)
                test_joint, test_pred = self._eval_step(data.test_x, data.test_y)

                m, sigma_bar = self.model.get("m", "sigma_bar")
                m, sigma_bar = m.cpu().numpy(), sigma_bar.cpu().numpy()
                measurements = np.where(m == 1)[0]

                log.info(f"val_joint {val_joint} test_joint {test_joint}")
                results[Ci] = dict(
                    val_joint=val_joint,
                    test_joint=test_joint,
                    measurements=measurements,
                    sigma_bar=sigma_bar,
                )
                if self.other_options.save_output:
                    results[Ci]["test_output"] = test_pred

        log.info("End of Training")
        return results
