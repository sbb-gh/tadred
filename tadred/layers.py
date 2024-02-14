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

import numpy as np
import torch

return_act_func: dict[str, torch.nn.Module] = dict(
    relu=torch.nn.ReLU(),
    sigmoid=torch.nn.Sigmoid(),
    identity=torch.nn.Identity(),
)


class FCN(torch.nn.Module):
    """Fully-connected/Dense network."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        inter_units: list[int],
        inter_act_fn: str = "relu",
        final_act_fn: str = "identity",
        inp_loss_affine_0: np.ndarray | None = None,
        out_loss_affine_0: np.ndarray | None = None,
    ):
        """
        Args:
        in_dim: Number of input units
        out_dim: Number of output units
        inter_units: Number of units for intermediate layers
        inter_act_fn: Activation function for intermediate layers
        final_act_fn: Actication function on last layers
        inp_loss_affine_0: Affine normalizer multiplier for input
        out_loss_affine_0: Affine normalizer multiplier for output
        """
        super().__init__()
        if inp_loss_affine_0 is None:
            inp_affine_0_prod = np.array(1)
        else:
            inp_affine_0_prod = inp_loss_affine_0
        if out_loss_affine_0 is None:
            out_affine_0_prod = np.array(1)
        else:
            out_affine_0_prod = out_loss_affine_0
        self.register_buffer("inp_affine_0_prod", torch.tensor(inp_affine_0_prod))
        self.register_buffer("out_affine_0_prod", torch.tensor(out_affine_0_prod))

        layers: list[torch.nn.Module] = []
        if len(inter_units) == 0:
            layers.append(torch.nn.Linear(in_dim, out_dim))
        elif inter_units[0] == -1:
            pass
        else:
            for i, num_outputs in enumerate(inter_units):
                if i == 0:
                    in_features = in_dim
                else:
                    in_features = inter_units[i - 1]

                layers.append(torch.nn.Linear(in_features, num_outputs))
                layers.append(return_act_func[inter_act_fn])

            layers.append(torch.nn.Linear(inter_units[-1], out_dim))

        layers = layers + [return_act_func[final_act_fn]]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.inp_affine_0_prod is not None:
            x = x / self.inp_affine_0_prod
        x = self.layers(x)
        if self.out_affine_0_prod is not None:
            x = x * self.out_affine_0_prod
        return x


class DownsamplingMultLayer(torch.nn.Module):
    """Subsampling layer.
    This also holds TADRED variables, TODO change and move to main trainer?
    """

    def __init__(self, n_features: int, train_x_median: np.ndarray):
        super().__init__()
        self.register_buffer("sigma", torch.zeros(n_features))
        self.register_buffer("sigma_mult", torch.tensor(1.0))
        self.register_buffer("m", torch.ones(n_features))
        self.register_buffer("sigma_bar", torch.ones(n_features))
        self.register_buffer("sigma_average", torch.zeros(n_features))
        self.register_buffer("train_x_median", torch.tensor(train_x_median))

    # TODO https://stackoverflow.com/questions/37031928/type-annotations-for-args-and-kwargs
    def assign(self, **kwargs):
        for key, val in kwargs.items():
            if not torch.is_tensor(val):
                val = torch.tensor(val)
            old_val = getattr(self, str(key))
            val = val.clone().to(old_val.device)
            setattr(self, key, val)

    def get(self, *args):
        ret = []
        for key in args:
            out = getattr(self, str(key))
            ret.append(out.clone())
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    def forward(self, x_inp, score_inp=1):
        if isinstance(score_inp, torch.Tensor):
            self.sigma = (torch.mean(score_inp, axis=0)).detach()
        score_tot = self.sigma_mult * score_inp + (1 - self.sigma_mult) * self.sigma_bar
        subsample = self.m * x_inp + (1 - self.m) * self.train_x_median
        out = score_tot * subsample
        return out


### Options for activation function


class Sigmoid(torch.nn.Module):
    def __init__(self, mult: int = 1):
        super().__init__()
        self.register_buffer("mult", torch.tensor(mult))

    def forward(self, x):
        return torch.sigmoid(x) * self.mult


class SigmoidRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_low = torch.clamp(x, min=None, max=0)
        y = torch.nn.functional.sigmoid(x_low) * 2
        x_high = torch.clamp(x, min=0, max=None)
        y = y + x_high
        return y


class Exp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


get_score_activation: dict[str, torch.nn.Module] = dict(
    doublesigmoid=Sigmoid(mult=2),
    sigmoidrelu=SigmoidRelu(),
    exp=Exp(),
    relu=torch.nn.ReLU(),
)
