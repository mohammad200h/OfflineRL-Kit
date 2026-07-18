import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional, Literal
from offlinerlkit.nets import EnsembleLinear


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x


def soft_clamp(
    x : torch.Tensor,
    _min: Optional[torch.Tensor] = None,
    _max: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


RewardMode = Literal["twohot", "gaussian_joint"]


class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        num_elites: int = 5,
        activation: nn.Module = Swish,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        with_reward: bool = True,
        reward_mode: RewardMode = "twohot",
        num_reward_bins: int = 255,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_reward = with_reward
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_mode = reward_mode
        self.num_reward_bins = num_reward_bins
        self.device = torch.device(device)

        self.activation = activation()

        assert len(weight_decays) == (len(hidden_dims) + 1)

        module_list = []
        hidden_dims = [obs_dim+action_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        for in_dim, out_dim, weight_decay in zip(hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1]):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
        self.backbones = nn.ModuleList(module_list)

        if reward_mode == "twohot":
            dynamics_out_dim = 2 * obs_dim
            self.output_layer = EnsembleLinear(
                hidden_dims[-1],
                dynamics_out_dim,
                num_ensemble,
                weight_decays[-1],
            )
            self.reward_head = EnsembleLinear(
                hidden_dims[-1],
                num_reward_bins,
                num_ensemble,
                weight_decays[-1],
            )
            logvar_dim = obs_dim
        else:
            dynamics_out_dim = 2 * (obs_dim + int(with_reward))
            self.output_layer = EnsembleLinear(
                hidden_dims[-1],
                dynamics_out_dim,
                num_ensemble,
                weight_decays[-1],
            )
            self.reward_head = None
            logvar_dim = obs_dim + int(with_reward)

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(logvar_dim) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(logvar_dim) * -10, requires_grad=True)
        )

        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        )

        self.to(self.device)

    def _shared_features(self, obs_action: torch.Tensor) -> torch.Tensor:
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        return output

    def forward(
        self, obs_action: Union[np.ndarray, torch.Tensor]
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        obs_action = torch.as_tensor(obs_action, dtype=torch.float32).to(self.device)
        features = self._shared_features(obs_action)
        if self.reward_mode == "twohot":
            mean, logvar = torch.chunk(self.output_layer(features), 2, dim=-1)
            logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
            reward_logits = self.reward_head(features)
            return mean, logvar, reward_logits
        mean, logvar = torch.chunk(self.output_layer(features), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()
        if self.reward_head is not None:
            self.reward_head.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)
        if self.reward_head is not None:
            self.reward_head.update_save(indexes)
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        if self.reward_head is not None:
            decay_loss += self.reward_head.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
    
    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs
