from __future__ import annotations

import inspect
from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
  # Bisa ditambahi lagi, but basically common interface/base class buat per algortimanya

  @classmethod
  def _get_param_names(cls) -> list[str]:
    """Read hyperparameter names from the __init__ signature."""
    if cls.__init__ is object.__init__:
      return []
    names = []
    for p in inspect.signature(cls.__init__).parameters.values():
      if p.name == "self":
          continue
      if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
          raise TypeError(
            f"{cls.__name__}.__init__ must not use *args or **kwargs"
          )
      names.append(p.name)
    return names

  def get_params(self, deep: bool = True) -> dict:
    """Return hyperparameters as a dict. Required for sklearn clone()."""
    return {name: getattr(self, name) for name in self._get_param_names()}

  def set_params(self, **params) -> BaseModel:
    """Set hyperparameters by name. Required by GridSearchCV"""
    valid = self._get_param_names()
    for name, value in params.items():
      if name not in valid:
        raise ValueError(
          f"Invalid parameter {name!r} for {type(self).__name__}"
          f"Valid parameters: {valid}"
        )
      setattr(self, name, value)
    return self

  def __repr__(self) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
    return f"{type(self).__name__}({args})"

  @abstractmethod
  def fit(self, X, y) -> BaseModel:
    pass

  def predict(self, X) -> np.ndarray:
    pass