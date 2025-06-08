from dataclasses import dataclass
from typing import List, Dict

import torch
import scipy as sp
import scipy.stats


@dataclass(frozen=True)
class OutputStatistics:
    empirical_mean: float
    empirical_variance: float
    empirical_kurtosis: float


class OutputStatisticsMeasurement:
    def __init__(self) -> None:
        self.layer_outputs: Dict[int, List[float]] = {}

    def add_layer_output(self, layer_output: torch.Tensor, layer_index: int) -> None:
        self.layer_outputs[layer_index] = self.layer_outputs.get(layer_index, [])
        self.layer_outputs[layer_index].extend(layer_output[:, 0].detach().cpu().tolist())

    def depths(self) -> List[int]:
        return sorted(list(self.layer_outputs.keys()))

    def empirical_variances(self) -> List[float]:
        depths = self.depths()
        empirical_variances = [sp.stats.tvar(self.layer_outputs[d]) for d in depths]
        return empirical_variances
