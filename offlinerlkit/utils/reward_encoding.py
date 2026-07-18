"""Symlog + twohot reward encoding (DreamerV3-style) and scaling utilities."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Literal, Tuple, Union

import numpy as np

RewardScaleMode = Literal["none", "divide", "standardize"]


def symlog(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def symexp(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.expm1(np.abs(x))


@dataclass
class RewardPreprocessor:
    """Affine map applied before symlog: r_norm = (r - bias) / scale."""

    scale: float = 1.0
    bias: float = 0.0
    mode: RewardScaleMode = "none"

    def transform(self, rewards: np.ndarray) -> np.ndarray:
        return (np.asarray(rewards, dtype=np.float64) - self.bias) / self.scale

    def inverse(self, rewards_norm: np.ndarray) -> np.ndarray:
        return np.asarray(rewards_norm, dtype=np.float64) * self.scale + self.bias

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RewardPreprocessor":
        return cls(
            scale=float(d.get("scale", 1.0)),
            bias=float(d.get("bias", 0.0)),
            mode=d.get("mode", "none"),
        )


def fit_reward_preprocessor(
    rewards: np.ndarray, task: str
) -> RewardPreprocessor:
    """Pick reward scaling to match env family."""
    task_l = task.lower()
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if "mountaincar" in task_l:
        return RewardPreprocessor(scale=100.0, bias=0.0, mode="divide")
    if "antmaze" in task_l or "maze2d" in task_l:
        return RewardPreprocessor(scale=1.0, bias=0.0, mode="none")
    std = float(rewards.std())
    if std < 1e-6:
        std = 1.0
    return RewardPreprocessor(
        scale=std, bias=float(rewards.mean()), mode="standardize"
    )


@dataclass
class SymlogTwoHotEncoder:
    """Fixed symlog-space bins with twohot targets and softmax decoding."""

    num_bins: int = 255
    symlog_min: float = -20.0
    symlog_max: float = 20.0

    def __post_init__(self) -> None:
        self.bin_centers = np.linspace(
            self.symlog_min, self.symlog_max, self.num_bins, dtype=np.float64
        )
        if self.num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        self.bin_width = float(self.bin_centers[1] - self.bin_centers[0])

    @property
    def symlog_range(self) -> Tuple[float, float]:
        return self.symlog_min, self.symlog_max

    def rewards_to_symlog(self, rewards_norm: np.ndarray) -> np.ndarray:
        return symlog(rewards_norm)

    def encode_twohot(self, rewards_norm: np.ndarray) -> np.ndarray:
        """Return (batch, num_bins) soft twohot targets in symlog space."""
        y = self.rewards_to_symlog(rewards_norm).reshape(-1)
        n = len(y)
        targets = np.zeros((n, self.num_bins), dtype=np.float64)
        pos = (y - self.bin_centers[0]) / self.bin_width
        pos = np.clip(pos, 0.0, self.num_bins - 1 - 1e-6)
        idx_lo = np.floor(pos).astype(np.int64)
        idx_hi = idx_lo + 1
        w_hi = pos - idx_lo
        w_lo = 1.0 - w_hi
        rows = np.arange(n)
        targets[rows, idx_lo] = w_lo
        targets[rows, idx_hi] = w_hi
        return targets.astype(np.float32)

    def primary_bin_index(self, rewards_norm: np.ndarray) -> np.ndarray:
        """Bin with largest twohot mass (for eval accuracy)."""
        twohot = self.encode_twohot(rewards_norm)
        return np.argmax(twohot, axis=-1).astype(np.int64)

    def decode_symlog_expectation(self, logits: np.ndarray) -> np.ndarray:
        """Decode logits (..., num_bins) to reward in normalized space."""
        logits = np.asarray(logits, dtype=np.float64)
        logits = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        symlog_val = np.tensordot(probs, self.bin_centers, axes=([-1], [0]))
        return symexp(symlog_val).astype(np.float32)

    def decode_bin_indices(self, logits: np.ndarray) -> np.ndarray:
        return np.argmax(logits, axis=-1).astype(np.int64)

    def cross_entropy(self, logits: np.ndarray, targets_twohot: np.ndarray) -> float:
        logits = np.asarray(logits, dtype=np.float64)
        targets = np.asarray(targets_twohot, dtype=np.float64)
        logits = logits - logits.max(axis=-1, keepdims=True)
        log_probs = logits - np.log(
            np.exp(logits).sum(axis=-1, keepdims=True) + 1e-8
        )
        ce = -(targets * log_probs).sum(axis=-1)
        return float(np.mean(ce))

    def to_dict(self) -> dict:
        return {
            "num_bins": self.num_bins,
            "symlog_min": self.symlog_min,
            "symlog_max": self.symlog_max,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SymlogTwoHotEncoder":
        return cls(
            num_bins=int(d.get("num_bins", 255)),
            symlog_min=float(d.get("symlog_min", -20.0)),
            symlog_max=float(d.get("symlog_max", 20.0)),
        )


@dataclass
class RewardEncodingConfig:
    reward_mode: Literal["twohot", "gaussian_joint"] = "twohot"
    preprocessor: RewardPreprocessor = None
    encoder: SymlogTwoHotEncoder = None
    reward_loss_weight: float = 1.0
    dynamics_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.preprocessor is None:
            self.preprocessor = RewardPreprocessor()
        if self.encoder is None:
            self.encoder = SymlogTwoHotEncoder()

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        payload = {
            "reward_mode": self.reward_mode,
            "reward_loss_weight": self.reward_loss_weight,
            "dynamics_loss_weight": self.dynamics_loss_weight,
            "preprocessor": self.preprocessor.to_dict(),
            "encoder": self.encoder.to_dict(),
        }
        path = os.path.join(directory, "reward_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "RewardEncodingConfig":
        path = os.path.join(directory, "reward_config.json")
        if not os.path.isfile(path):
            return cls(reward_mode="gaussian_joint")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            reward_mode=payload.get("reward_mode", "twohot"),
            preprocessor=RewardPreprocessor.from_dict(payload["preprocessor"]),
            encoder=SymlogTwoHotEncoder.from_dict(payload["encoder"]),
            reward_loss_weight=float(payload.get("reward_loss_weight", 1.0)),
            dynamics_loss_weight=float(payload.get("dynamics_loss_weight", 1.0)),
        )

    def encode_rewards(self, rewards: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (rewards_norm, twohot targets)."""
        rewards_norm = self.preprocessor.transform(rewards)
        twohot = self.encoder.encode_twohot(rewards_norm)
        return rewards_norm.astype(np.float32), twohot

    def decode_logits_to_normalized(self, logits: np.ndarray) -> np.ndarray:
        return self.encoder.decode_symlog_expectation(logits)

    def decode_logits_to_raw(self, logits: np.ndarray) -> np.ndarray:
        norm = self.decode_logits_to_normalized(logits)
        return self.preprocessor.inverse(norm)
