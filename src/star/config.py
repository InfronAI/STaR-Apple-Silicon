"""Explicit, validated configuration shared by generation, training and evaluation."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    model: str = "openai-community/gpt2"
    revision: str = "main"
    data_root: str = "datasets"
    output_dir: str = "runs/commonsenseqa"
    task: str = "commonsenseqa"
    train_file: str | None = None
    eval_file: str | None = None
    device: str = "auto"
    local_files_only: bool = False
    seed: int = 42
    iterations: int = 2
    train_samples: int = 32
    eval_samples: int = 16
    few_shot: int = 3
    few_shot_train: bool = False
    direct: bool = False
    rationalize: bool = True
    rationalization_probability: float = 1.0
    max_seq_length: int = 512
    max_new_tokens: int = 96
    generation_batch_size: int = 1
    temperature: float = 0.0
    top_p: float = 0.9
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    epochs: float = 1.0
    add_epochs: float = 0.2
    max_steps: int | None = None
    add_steps: int = 0
    loss_mask: str = "completion"
    z_loss: float = 1e-4

    def validate(self):
        self.task = {"cqa": "commonsenseqa", "gsm8k": "gsm"}.get(self.task, self.task)
        if self.task not in {"commonsenseqa", "gsm", "arithmetic"}:
            raise ValueError("task must be commonsenseqa, gsm, or arithmetic")
        if self.device not in {"auto", "mps", "cpu"}:
            raise ValueError("device must be auto, mps, or cpu")
        if self.loss_mask not in {"completion", "all"}:
            raise ValueError("loss_mask must be completion or all")
        for name in (
            "iterations",
            "train_samples",
            "eval_samples",
            "max_seq_length",
            "max_new_tokens",
            "generation_batch_size",
            "batch_size",
            "gradient_accumulation_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("few_shot", "add_steps", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "local_files_only",
            "few_shot_train",
            "direct",
            "rationalize",
            "gradient_checkpointing",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("learning_rate", "epochs", "max_grad_norm", "top_p"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "temperature",
            "weight_decay",
            "add_epochs",
            "z_loss",
            "rationalization_probability",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.top_p > 1 or self.rationalization_probability > 1:
            raise ValueError("top_p and rationalization_probability must be <= 1")
        if self.max_new_tokens >= self.max_seq_length:
            raise ValueError("max_new_tokens must be smaller than max_seq_length")
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps < 1):
            raise ValueError("max_steps must be null or a positive integer")
        if not self.model or not self.revision:
            raise ValueError("model and revision cannot be empty")
        for name in ("model", "revision", "data_root", "output_dir"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        for name in ("train_file", "eval_file"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be null or a string")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, filename):
        path = Path(filename).expanduser().resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Config must be a JSON object")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
        config = cls(**data).validate()
        for name in ("data_root", "output_dir", "train_file", "eval_file"):
            value = getattr(config, name)
            if value is not None:
                setattr(config, name, str((path.parent / Path(value).expanduser()).resolve()))
        if config.model.startswith((".", "/", "~")):
            config.model = str((path.parent / Path(config.model).expanduser()).resolve())
        return config
