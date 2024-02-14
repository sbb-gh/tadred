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

import numpy as np
import torch
from omegaconf import OmegaConf

from .layers import FCN, DownsamplingMultLayer, get_score_activation
from .types import TadredTrainEval

log = logging.getLogger(__name__)


class TADREDBase(torch.nn.Module):
    """Super class for TADRED-specific optimization/training."""

    def __init__(
        self,
        feature_set_sizes_Ci: list[int],
        feature_set_sizes_evaluated: list[int],
        epochs: int,
        epochs_decay: int,
        epochs_fix_sigma: int,
        epochs_decay_sigma: int,
    ):
        """Arguments from paper, see tadred_train_eval in config file."""
        super().__init__()
        self.feature_set_sizes_Ci = feature_set_sizes_Ci
        self.n_features = self.feature_set_sizes_Ci[0]

        self.epochs_decay = epochs_decay
        self.alpha_m = 1.0 / epochs_decay

        self.epochs_fix_sigma = epochs_fix_sigma
        self.epochs_decay_sigma = epochs_decay_sigma
        self.alpha_sigma = 0.5 / epochs_decay_sigma
        assert epochs_fix_sigma + epochs_decay_sigma + epochs_decay < epochs
        if not set(feature_set_sizes_evaluated).issubset(feature_set_sizes_Ci):
            log.warning(
                f"Set feature_set_sizes_evaluated {feature_set_sizes_evaluated} "
                f"as subset of feature_set_sizes_Ci {feature_set_sizes_Ci}"
            )

        self.t = 0  # Time step of outer loop

    def assign(self, **kwargs):
        self.downsampling_mult_layer.assign(**kwargs)

    def get(self, *args):
        return self.downsampling_mult_layer.get(*args)

    def on_step_begin(self):
        """On step t=1,..,T set features to remove, update sigma_mult"""
        self.t += 1
        self.epoch = 0
        m = self.get("m")
        log.info(
            f"m has {len(torch.where(m == 1)[0])} ones and {len(torch.where(m == 0)[0])} zeros"
        )
        # Number of measurements to remove this step
        if self.t == 1:
            self.Dt = self.n_features - self.feature_set_sizes_Ci[0]
            sigma_mult = 1
        else:
            self.Dt = self.feature_set_sizes_Ci[self.t - 2] - self.feature_set_sizes_Ci[self.t - 1]
            sigma_mult = 0.5
        self.assign(sigma_mult=sigma_mult)
        log.info(f"sigma_mult {sigma_mult}")
        self._set_m_decay()

    def on_epoch_begin(self):
        """At beginning of epoch set sigma_bar, sigma_mult, mask m, depending on phase"""
        self.epoch += 1
        m, sigma_mult = self.get("m", "sigma_mult")
        self.sigma_average_list = []
        self.no_batches = 0

        if self.t == 0:
            return

        if self.epoch == self.epochs_fix_sigma:
            log.info("Trigger epochs_fix_sigma")
            sigma_average, sigma_bar = self.get("sigma_average", "sigma_bar")
            sigma_bar = 0.5 * (sigma_bar + sigma_average)
            self.assign(sigma_bar=sigma_bar)
            log.info("sigma_bar = 0.5*(sigma_bar+sigma_average)")

        if self.epoch >= self.epochs_fix_sigma:
            if sigma_mult > 0:
                sigma_mult = sigma_mult - self.alpha_sigma
                sigma_mult = torch.max(sigma_mult, torch.tensor(0).type_as(sigma_mult))
                log.info(f"Decay sigma_mult {float(sigma_mult)}")
                self.assign(sigma_mult=sigma_mult)

        if self.epoch >= (self.epochs_fix_sigma + self.epochs_decay_sigma):
            if torch.sum(self.m_decay > 0) and torch.max(m[torch.where(self.m_decay == 1)]) > 0:
                log.info("Decay measurements")
                m = m - self.alpha_m * self.m_decay
                m = torch.max(m, torch.tensor(0).type_as(m))
                self.assign(m=m)

    def on_batch_begin_train(self):
        pass

    def on_batch_end_train(self):
        """Cache mean of learnt scores across batch"""
        self.no_batches += 1
        sigma = self.get("sigma")
        self.sigma_average_list.append(sigma)

    def on_epoch_end(self):
        """Compute averaged score across all data"""
        sigma_average_list = torch.stack(self.sigma_average_list)
        sigma_average_list = torch.mean(sigma_average_list, axis=0)
        self.assign(sigma_average=sigma_average_list)

    def on_step_end(self):
        """Call at the end of step t=1,...,T to cache averaged score."""
        if self.t == 1:
            self.assign(sigma_bar=self.get("sigma_average"))

    def _set_m_decay(self):
        """Choose and set features to remove m_{t-1}-m_{t}."""
        m, sigma_bar = self.get("m", "sigma_bar")
        self.m_decay = torch.zeros(self.n_features, dtype=torch.int).to(m.device)
        # Decay self.D[t] measurements in NAS step t, Dt in paper
        if self.Dt > 0:
            # Find indices smallest sigma_bar where m!=0
            assert torch.sum((0 < m) & (m < 1)) == 0, "m has values between 0,1"

            m_decay_options = torch.where(m == 1)
            m_decay_options_sigma = sigma_bar[m_decay_options]
            m_decay_options_sigma = torch.argsort(m_decay_options_sigma)[: self.Dt]
            D = m_decay_options[0][m_decay_options_sigma]
            self.m_decay[D] = 1
        log.info(f"Decay: {float(torch.sum(self.m_decay))} measurements in NAS step {self.t}")


# TODO -- should this be composition instead of inheritance, also remove base class
class TADREDNet(TADREDBase):
    """TADRED Network"""

    def __init__(
        self,
        num_units_score: list[int],
        num_units_task: list[int],
        score_activation: str,
        n_features: int,
        out_units: int,
        train_x_median: np.ndarray,
        loss_affine_x: tuple[np.ndarray, np.ndarray],
        loss_affine_y: tuple[np.ndarray, np.ndarray],
        tadred_train_eval: TadredTrainEval,
    ):
        """Define Scoring and Prediction Networks with forward pass.

        Args:
        num_units_score, num_units_task, score_activation: Config file network parameters
        n_features, out_units, train_x_median, loss_affine_x, loss_affine_y: Data features
        tadred_train_eval: Config file tadred_train_eval parameters
        """
        super().__init__(**OmegaConf.to_container(tadred_train_eval))
        self.score_net = FCN(
            in_dim=n_features,
            out_dim=n_features,
            inter_units=num_units_score,
            inter_act_fn="relu",
            final_act_fn="identity",
            inp_loss_affine_0=loss_affine_x[0],
            out_loss_affine_0=None,
        )

        self.task_net = FCN(
            in_dim=n_features,
            out_dim=out_units,
            inter_units=num_units_task,
            inter_act_fn="relu",
            final_act_fn="identity",
            inp_loss_affine_0=loss_affine_x[0],
            out_loss_affine_0=loss_affine_y[0],
        )

        self.score_activation = get_score_activation[score_activation]
        self.downsampling_mult_layer = DownsamplingMultLayer(
            n_features=n_features,
            train_x_median=train_x_median,
        )
        self.loss_fct = torch.nn.MSELoss()

    def forward_score(self, x_inp):
        score = self.score_net(x_inp)
        score = self.score_activation(score)
        return score

    def forward_eval(self, x_inp, score):
        x_subsampled_weighted = self.downsampling_mult_layer(x_inp, score)
        x = self.task_net(x_subsampled_weighted)
        return x

    def forward(self, x):
        score = self.forward_score(x)
        y = self.forward_eval(x, score)
        return y

    def forward_and_loss(self, x, y):
        y_hat = self.forward(x)
        loss = self.loss_fct(y, y_hat)
        return loss

    def forward_and_backward(self, x, y):
        loss = self.forward_and_loss(x, y)
        loss.backward()
        return loss
