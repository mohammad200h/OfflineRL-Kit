import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
from offlinerlkit.utils.reward_encoding import (
    RewardEncodingConfig,
    fit_reward_preprocessor,
)


class EnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
        reward_config: Optional[RewardEncodingConfig] = None,
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
        self.reward_config = reward_config or RewardEncodingConfig()
        self._obs_dim = int(model.obs_dim)

    @property
    def _twohot(self) -> bool:
        return (
            self.reward_config.reward_mode == "twohot"
            and getattr(self.model, "reward_mode", "gaussian_joint") == "twohot"
        )

    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        if self._twohot:
            delta_mean, delta_logvar, reward_logits = self.model(obs_act)
            delta_mean = delta_mean.cpu().numpy()
            delta_logvar = delta_logvar.cpu().numpy()
            reward_logits = reward_logits.cpu().numpy()
            delta_std = np.sqrt(np.exp(delta_logvar))
            delta_samples = (
                delta_mean + np.random.normal(size=delta_mean.shape) * delta_std
            ).astype(np.float32)
            num_models, batch_size, _ = delta_samples.shape
            model_idxs = self.model.random_elite_idxs(batch_size)
            delta = delta_samples[model_idxs, np.arange(batch_size)]
            next_obs = obs + delta
            elite_logits = reward_logits[self.model.elites.data.cpu().numpy()]
            mean_logits = elite_logits.mean(axis=0)
            reward_norm = self.reward_config.encoder.decode_symlog_expectation(
                mean_logits
            )
            reward = np.asarray(reward_norm, dtype=np.float32).reshape(-1, 1)
            std = delta_std
        else:
            mean, logvar = self.model(obs_act)
            mean = mean.cpu().numpy()
            logvar = logvar.cpu().numpy()
            mean[..., :-1] += obs
            std = np.sqrt(np.exp(logvar))
            ensemble_samples = (
                mean + np.random.normal(size=mean.shape) * std
            ).astype(np.float32)
            num_models, batch_size, _ = ensemble_samples.shape
            model_idxs = self.model.random_elite_idxs(batch_size)
            samples = ensemble_samples[model_idxs, np.arange(batch_size)]
            next_obs = samples[..., :-1]
            reward = samples[..., -1:]
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward

        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                if self._twohot:
                    next_obses_mean = obs[None, :, :] + delta_mean[..., : self._obs_dim]
                else:
                    next_obses_mean = mean[..., :-1]
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                if self._twohot:
                    next_obses_mean = obs[None, :, :] + delta_mean[..., : self._obs_dim]
                else:
                    next_obses_mean = mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty
        
        return next_obs, reward, terminal, info
    
    @ torch.no_grad()
    def sample_next_obss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=-1)
        obs_act = self.scaler.transform_tensor(obs_act)
        if self._twohot:
            delta_mean, delta_logvar, _ = self.model(obs_act)
            obs_expanded = obs.unsqueeze(0).expand_as(delta_mean)
            mean = delta_mean + obs_expanded
            std = torch.sqrt(torch.exp(delta_logvar))
        else:
            mean, logvar = self.model(obs_act)
            mean[..., :-1] += obs
            std = torch.sqrt(torch.exp(logvar))

        mean = mean[self.model.elites.data.cpu().numpy()]
        std = std[self.model.elites.data.cpu().numpy()]

        samples = torch.stack(
            [mean + torch.randn_like(std) * std for _ in range(num_samples)], 0
        )
        if self._twohot:
            next_obss = samples[..., : self._obs_dim]
        else:
            next_obss = samples[..., :-1]
        return next_obss

    def format_samples_for_training(
        self, data: Dict
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        if self._twohot:
            _, reward_twohot = self.reward_config.encode_rewards(rewards)
            return inputs, delta_obss.astype(np.float32), reward_twohot
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        return inputs, targets, None

    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01,
        eval_callback: Optional[Callable[["EnsembleDynamics", int], Optional[Dict]]] = None,
        eval_freq: int = 1,
    ) -> None:
        if self._twohot and "rewards" in data:
            self.reward_config.preprocessor = fit_reward_preprocessor(
                data["rewards"], data.get("task", "")
            )

        inputs, train_targets, reward_twohot = self.format_samples_for_training(data)
        if not self._twohot:
            targets = train_targets
        else:
            targets = train_targets
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(
            range(data_size), (train_size, holdout_size)
        )
        train_inputs = inputs[train_splits.indices]
        holdout_inputs = inputs[holdout_splits.indices]
        if self._twohot:
            train_delta = targets[train_splits.indices]
            holdout_delta = targets[holdout_splits.indices]
            train_reward = reward_twohot[train_splits.indices]
            holdout_reward = reward_twohot[holdout_splits.indices]
        else:
            train_delta = targets[train_splits.indices]
            holdout_delta = targets[holdout_splits.indices]
            train_reward = None
            holdout_reward = None

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_losses = [1e10 for _ in range(self.model.num_ensemble)]

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])

        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            train_loss = self.learn(
                train_inputs[data_idxes],
                train_delta[data_idxes],
                train_reward[data_idxes] if self._twohot else None,
                batch_size,
                logvar_loss_coef,
            )
            new_holdout_losses = self.validate(
                holdout_inputs,
                holdout_delta,
                holdout_reward,
            )
            holdout_loss = (np.sort(new_holdout_losses)[: self.model.num_elites]).mean()
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)

            if eval_callback is not None and eval_freq > 0 and (epoch % eval_freq) == 0:
                eval_metrics = eval_callback(self, epoch)
                if eval_metrics:
                    for key, value in eval_metrics.items():
                        logger.logkv(key, value)

            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(
                range(len(holdout_losses)), new_holdout_losses, holdout_losses
            ):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss

            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1

            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        self.save(logger.model_dir)
        self.model.eval()
        logger.log(
            "elites:{} , holdout loss: {}".format(
                indexes, (np.sort(holdout_losses)[: self.model.num_elites]).mean()
            )
        )

        if eval_callback is not None:
            eval_metrics = eval_callback(self, epoch)
            if eval_metrics:
                for key, value in eval_metrics.items():
                    logger.logkv(key, value)
                logger.set_timestep(epoch)
                logger.dumpkvs(exclude=["policy_training_progress"])

    def _dynamics_gaussian_loss(
        self,
        delta_mean: torch.Tensor,
        delta_logvar: torch.Tensor,
        delta_targets: torch.Tensor,
        logvar_loss_coef: float,
    ) -> torch.Tensor:
        inv_var = torch.exp(-delta_logvar)
        mse_loss_inv = (torch.pow(delta_mean - delta_targets, 2) * inv_var).mean(
            dim=(1, 2)
        )
        var_loss = delta_logvar.mean(dim=(1, 2))
        loss = mse_loss_inv + var_loss
        loss = loss.sum()
        loss = loss + self.model.get_decay_loss()
        loss = loss + logvar_loss_coef * self.model.max_logvar.sum()
        loss = loss - logvar_loss_coef * self.model.min_logvar.sum()
        return loss

    def _reward_twohot_loss(
        self, reward_logits: torch.Tensor, reward_targets: torch.Tensor
    ) -> torch.Tensor:
        log_probs = F.log_softmax(reward_logits, dim=-1)
        ce = -(reward_targets * log_probs).sum(dim=-1).mean(dim=-1)
        return ce.sum()

    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        reward_targets: Optional[np.ndarray],
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01,
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []
        dyn_w = self.reward_config.dynamics_loss_weight
        rew_w = self.reward_config.reward_loss_weight

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            sl = slice(batch_num * batch_size, (batch_num + 1) * batch_size)
            inputs_batch = inputs[:, sl]
            targets_batch = targets[:, sl]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)

            if self._twohot:
                reward_batch = torch.as_tensor(reward_targets[:, sl]).to(
                    self.model.device
                )
                delta_mean, delta_logvar, reward_logits = self.model(inputs_batch)
                loss = dyn_w * self._dynamics_gaussian_loss(
                    delta_mean, delta_logvar, targets_batch, logvar_loss_coef
                )
                loss = loss + rew_w * self._reward_twohot_loss(
                    reward_logits, reward_batch
                )
            else:
                mean, logvar = self.model(inputs_batch)
                inv_var = torch.exp(-logvar)
                mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(
                    dim=(1, 2)
                )
                var_loss = logvar.mean(dim=(1, 2))
                loss = mse_loss_inv.sum() + var_loss.sum()
                loss = loss + self.model.get_decay_loss()
                loss = loss + logvar_loss_coef * self.model.max_logvar.sum()
                loss = loss - logvar_loss_coef * self.model.min_logvar.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.append(loss.item())
        return np.mean(losses)

    @ torch.no_grad()
    def validate(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        reward_targets: Optional[np.ndarray] = None,
    ) -> List[float]:
        self.model.eval()
        targets_t = torch.as_tensor(targets).to(self.model.device)
        dyn_w = self.reward_config.dynamics_loss_weight
        rew_w = self.reward_config.reward_loss_weight

        if self._twohot:
            reward_t = torch.as_tensor(reward_targets).to(self.model.device)
            delta_mean, delta_logvar, reward_logits = self.model(inputs)
            delta_mse = ((delta_mean - targets_t) ** 2).mean(dim=(1, 2))
            log_probs = F.log_softmax(reward_logits, dim=-1)
            reward_ce = -(reward_t * log_probs).sum(dim=-1).mean(dim=-1)
            loss = dyn_w * delta_mse + rew_w * reward_ce
            return list(loss.cpu().numpy())

        mean, _ = self.model(inputs)
        loss = ((mean - targets_t) ** 2).mean(dim=(1, 2))
        return list(loss.cpu().numpy())

    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
        self.reward_config.save(save_path)

    def load(self, load_path: str) -> None:
        self.model.load_state_dict(
            torch.load(
                os.path.join(load_path, "dynamics.pth"),
                map_location=self.model.device,
            )
        )
        self.scaler.load_scaler(load_path)
        loaded = RewardEncodingConfig.load(load_path)
        if loaded.reward_mode != "gaussian_joint":
            self.reward_config = loaded
