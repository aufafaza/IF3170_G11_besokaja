from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
  # Bisa ditambahi lagi, but basically common interface/base class buat per algortimanya

  @abstractmethod
  def fit(self, X, y) -> BaseModel:
    pass

  def predict(self, X) -> np.ndarray:
    pass

  