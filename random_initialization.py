try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol
import torch
import torch.nn as nn


class ZeroMeanInitializer(Protocol):
    def initialize(self, weight_matrix: torch.Tensor, std: float):
        raise NotImplementedError()


class NormalZeroMeanInitializer(ZeroMeanInitializer):
    def initialize(self, weight_matrix: torch.Tensor, std: float) -> None:
        nn.init.normal_(weight_matrix, mean=0., std=std)


class UniformZeroMeanInitializer(ZeroMeanInitializer):
    def initialize(self, weight_matrix: torch.Tensor, std: float) -> None:
        a = (3 * std**2)**0.5
        nn.init.uniform_(weight_matrix, a=-a, b=a)


class BernoulliZeroMeanInitializer(ZeroMeanInitializer):
    def initialize(self, weight_matrix: torch.Tensor, std: float) -> None:
        with torch.no_grad():
            tensor = weight_matrix.uniform_()
            positive = tensor > 0.5
            negative = tensor <= 0.5
            tensor[negative] = -std
            tensor[positive] = std
