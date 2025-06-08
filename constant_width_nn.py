from typing import Optional, Callable, Tuple
import numpy as np
import itertools

import torch
import torch.nn as nn

from output_statistics import OutputStatisticsMeasurement


class ConstantWidthNN(nn.Module):
    def __init__(self,
                 depth: int,
                 width: int,
                 input_shape: Tuple[int, ...],
                 out_width: int,
                 bias: bool,
                 conv: bool = False,
                 normalization_factory: Optional[Callable[[int], nn.Module]] = None,
                 activation_factory: Optional[Callable[[], nn.Module]] = None,
                 ):
        super(ConstantWidthNN, self).__init__()
        if depth < 0:
            raise ValueError("Cannot initialize a network of negative depths")
        activation_factory = activation_factory or nn.ReLU
        normalization_factory = normalization_factory or (lambda _: nn.Identity())

        if conv:
            assert len(input_shape) == 3
            input_width = input_shape[0]
            linear_layer_factory = lambda in_ch, out_ch, b: nn.Conv2d(in_channels=in_ch, out_channels=out_ch, bias=b, kernel_size=3, padding=1)
            self.preprocess = nn.Identity()
        else:
            input_width = np.prod(input_shape)
            linear_layer_factory = lambda in_f, out_f, b: nn.Linear(in_features=in_f, out_features=out_f, bias=b)
            self.preprocess = nn.Flatten()

        layers = []
        current_width = input_width
        for i in range(depth):
            layers.append(linear_layer_factory(current_width, width, bias))
            layers.append(normalization_factory(width))
            layers.append(activation_factory())
            current_width = width

        if conv:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            layers.append(nn.Conv2d(in_channels=current_width, out_channels=out_width, bias=bias, kernel_size=1))
        else:
            layers.append(nn.Linear(in_features=current_width, out_features=out_width, bias=bias))
        self.input_layer = layers[0]
        self.layers = nn.ModuleList(layers[1:])
        self.statistics_measurement: Optional[OutputStatisticsMeasurement] = None

    def set_statistics_measurement(self, statistics_measurement: Optional[OutputStatisticsMeasurement]) -> None:
        self.statistics_measurement = statistics_measurement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.preprocess(x)
        linear_layer_index = 0
        for layer in itertools.chain([self.input_layer], self.layers):
            y = layer(y)
            if (isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)) and self.statistics_measurement is not None:
                self.statistics_measurement.add_layer_output(layer_output=y, layer_index=linear_layer_index)
                linear_layer_index += 1
        if len(y.shape) == 4:
            y = torch.flatten(y, start_dim=1)
        return y
