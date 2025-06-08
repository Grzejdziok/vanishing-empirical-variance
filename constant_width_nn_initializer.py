try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol
from typing import Union, Tuple, Callable
import math

from scipy.linalg import hadamard
import torch
import torch.nn as nn

from constant_width_nn import ConstantWidthNN
from random_initialization import ZeroMeanInitializer


def _get_input_width(layer: Union[nn.Linear, nn.Conv2d]) -> int:
    if isinstance(layer, nn.Linear):
        return layer.in_features
    else:
        return layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]


def _get_output_width(layer: Union[nn.Linear, nn.Conv2d]) -> int:
    if isinstance(layer, nn.Linear):
        return layer.out_features
    else:
        return layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]


class ConstantWidthNNInitializer(Protocol):
    def initialize(self, model: ConstantWidthNN) -> None:
        raise NotImplementedError()


class HeInitializer(ConstantWidthNNInitializer):
    def __init__(self, weight_initializer: ZeroMeanInitializer) -> None:
        self.weight_initializer = weight_initializer

    def _initialize(self, model: ConstantWidthNN) -> None:
        self.weight_initializer.initialize(weight_matrix=model.input_layer.weight, std=(2 / _get_input_width(model.input_layer)) ** 0.5)
        if model.input_layer.bias is not None:
            nn.init.zeros_(model.input_layer.bias)

        for child in model.layers.modules():
            if isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d):
                self.weight_initializer.initialize(weight_matrix=child.weight, std=(2 / _get_input_width(child)) ** 0.5)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)

    def initialize(self, model: ConstantWidthNN) -> None:
        with torch.no_grad():
            self._initialize(model)


class GlorotInitializer(ConstantWidthNNInitializer):
    def __init__(self, weight_initializer: ZeroMeanInitializer) -> None:
        self.weight_initializer = weight_initializer

    def _initialize(self, model: ConstantWidthNN) -> None:
        self.weight_initializer.initialize(weight_matrix=model.input_layer.weight, std=(2 / (_get_input_width(model.input_layer) + _get_output_width(model.input_layer))) ** 0.5)
        if model.input_layer.bias is not None:
            nn.init.zeros_(model.input_layer.bias)

        for child in model.layers.modules():
            if isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d):
                self.weight_initializer.initialize(weight_matrix=child.weight, std=(2 / (_get_input_width(child) + _get_output_width(child))) ** 0.5)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)

    def initialize(self, model: ConstantWidthNN) -> None:
        with torch.no_grad():
            self._initialize(model)


class LeCunInitializer(ConstantWidthNNInitializer):
    def __init__(self, weight_initializer: ZeroMeanInitializer) -> None:
        self.weight_initializer = weight_initializer

    def _initialize(self, model: ConstantWidthNN) -> None:
        self.weight_initializer.initialize(weight_matrix=model.input_layer.weight, std=(1 / _get_input_width(model.input_layer)) ** 0.5)
        if model.input_layer.bias is not None:
            nn.init.zeros_(model.input_layer.bias)

        for child in model.layers.modules():
            if isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d):
                self.weight_initializer.initialize(weight_matrix=child.weight, std=(1 / _get_input_width(child)) ** 0.5)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)

    def initialize(self, model: ConstantWidthNN) -> None:
        with torch.no_grad():
            self._initialize(model)


class OrthogonalInitializer(ConstantWidthNNInitializer):
    def _initialize(self, model: ConstantWidthNN) -> None:
        for child in model.modules():
            if isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d):
                nn.init.orthogonal_(child.weight, gain=nn.init.calculate_gain(nonlinearity='relu'))
                if child.bias is not None:
                    nn.init.zeros_(child.bias)

    def initialize(self, model: ConstantWidthNN) -> None:
        with torch.no_grad():
            self._initialize(model)


# This function is adapted from: https://github.com/jiaweizzhao/ZerO-initialization/blob/main/example_mnist.ipynb
# Copyright [2021] [Jiawei Zhao]
def ZerO_Init_on_matrix(matrix_tensor: torch.Tensor) -> torch.Tensor:
    # Algorithm 1 in the paper.

    m = matrix_tensor.size(0)
    n = matrix_tensor.size(1)

    if m <= n:
        init_matrix = torch.nn.init.eye_(torch.empty(m, n))
    else:
        assert m > n
        clog_m = math.ceil(math.log2(m))
        p = 2 ** (clog_m)
        init_matrix = torch.nn.init.eye_(torch.empty(m, p)) @ (
                    torch.tensor(hadamard(p)).float() / (2 ** (clog_m / 2))) @ torch.nn.init.eye_(torch.empty(p, n))

    return init_matrix


class ZerOInitializer(ConstantWidthNNInitializer):

    def __init__(self, first_layer_random: bool = False) -> None:
        self._first_layer_random = first_layer_random

    def _initialize(self, model: ConstantWidthNN):
        for child in model.modules():
            if isinstance(child, nn.Linear):
                child.weight.data = ZerO_Init_on_matrix(child.weight.data)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
            elif isinstance(child, nn.Conv2d):
                raise ValueError("ZerO initialization is not supported yet for convolutional layers")
        if self._first_layer_random:
            nn.init.normal_(model.input_layer.weight, std=(1 / _get_input_width(model.input_layer)) ** 0.5)
            if model.input_layer.bias is not None:
                nn.init.zeros_(model.input_layer.bias)

    def initialize(self, model: ConstantWidthNN) -> None:
        with torch.no_grad():
            self._initialize(model)


class IdentityInitializer(ConstantWidthNNInitializer):

    def _initialize(self, model: ConstantWidthNN) -> None:
        for child in model.modules():
            if isinstance(child, nn.Linear):
                nn.init.eye_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
            elif isinstance(child, nn.Conv2d):
                nn.init.dirac_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)

    def initialize(self, model: ConstantWidthNN) -> None:
        with torch.no_grad():
            self._initialize(model)


class MinimalDependenceInitializer(ConstantWidthNNInitializer):
    def _initialize(self, model: ConstantWidthNN) -> None:
        nn.init.normal_(model.input_layer.weight, std=(1/_get_input_width(model.input_layer))**0.5)
        if model.input_layer.bias is not None:
            nn.init.zeros_(model.input_layer.bias)
        for child in model.layers.modules():
            if isinstance(child, nn.Linear):
                nn.init.eye_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
            elif isinstance(child, nn.Conv2d):
                nn.init.dirac_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)

    def initialize(self, model: ConstantWidthNN) -> None:
        with torch.no_grad():
            self._initialize(model)


# Delta-orthogonal initialization is adapted from https://github.com/yl-1993/ConvDeltaOrthogonal-Init/blob/master/_ext/nn/init.py
# the code is on MIT license
def conv_delta_orthogonal_(tensor, gain=1.):
    r"""Initializer that generates a delta orthogonal kernel for ConvNets.
    The shape of the tensor must have length 3, 4 or 5. The number of input
    filters must not exceed the number of output filters. The center pixels of the
    tensor form an orthogonal matrix. Other pixels are set to be zero. See
    algorithm 2 in [Xiao et al., 2018]: https://arxiv.org/abs/1806.05393
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`3 \leq n \leq 5`
        gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
    Examples:
        >>> w = torch.empty(5, 4, 3, 3)
        >>> nn.init.conv_delta_orthogonal_(w)
    """
    if tensor.ndimension() < 3 or tensor.ndimension() > 5:
        raise ValueError("The tensor to initialize must be at least "
                         "three-dimensional and at most five-dimensional")

    if tensor.size(1) > tensor.size(0):
        raise ValueError("In_channels cannot be greater than out_channels.")

    # Generate a random matrix
    a = tensor.new(tensor.size(0), tensor.size(0)).normal_(0, 1)
    # Compute the qr factorization
    q, r = torch.qr(a)
    # Make Q uniform
    d = torch.diag(r, 0)
    q *= d.sign()
    q = q[:, :tensor.size(1)]
    with torch.no_grad():
        tensor.zero_()
        if tensor.ndimension() == 3:
            tensor[:, :, (tensor.size(2) - 1) // 2] = q
        elif tensor.ndimension() == 4:
            tensor[:, :, (tensor.size(2) - 1) // 2, (tensor.size(3) - 1) // 2] = q
        else:
            tensor[:, :, (tensor.size(2) - 1) // 2, (tensor.size(3) - 1) // 2, (tensor.size(4) - 1) // 2] = q
        tensor.mul_(math.sqrt(gain))
    return tensor


class DeltaOrthogonalInitializer(ConstantWidthNNInitializer):
    def _initialize(self, model: ConstantWidthNN) -> None:
        if not isinstance(model.input_layer, nn.Conv2d):
            raise ValueError("Delta-orthogonal initialization supports only convolutional networks")
        conv_delta_orthogonal_(model.input_layer.weight, nn.init.calculate_gain("relu"))
        if model.input_layer.bias is not None:
            nn.init.zeros_(model.input_layer.bias)
        for child in model.layers.modules():
            if isinstance(child, nn.Linear):
                raise ValueError("Delta-orthogonal initialization supports only convolutional networks")
            elif isinstance(child, nn.Conv2d):
                conv_delta_orthogonal_(child.weight, nn.init.calculate_gain("relu"))
                if child.bias is not None:
                    nn.init.zeros_(child.bias)

    def initialize(self, model: ConstantWidthNN) -> None:
        with torch.no_grad():
            self._initialize(model)


# MetaInit initialization is adapted from https://github.com/diggerdu/MetaInit/blob/master/meta_init.py
# the code is on MIT license
def gradient_quotient(loss, params, eps=1e-5):
    grad = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)
    prod = torch.autograd.grad(sum([(g**2).sum() / 2 for g in grad]),
        params, retain_graph=True, create_graph=True)
    out = sum([((g - p) / (g + eps * (2*(g >= 0).float() - 1).detach()) - 1).abs().sum() for g, p in zip(grad, prod)])
    return out / sum([p.data.nelement() for p in params])


def metainit(model, criterion, x_size, y_size, lr=0.1, momentum=0.9, steps=500, eps=1e-5):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad and len(p.size()) >= 2]
    memory = [0] * len(params)
    final_gq = math.inf
    for i in range(steps):
        input = torch.Tensor(*x_size).normal_(0, 1)
        target = torch.randint(0, y_size, (x_size[0],))
        loss = criterion(model(input), target)
        gq = gradient_quotient(loss, list(model.parameters()), eps)

        grad = torch.autograd.grad(gq, params)
        for j, (p, g_all) in enumerate(zip(params, grad)):
            norm = p.data.norm().item()
            g = torch.sign((p.data * g_all).sum() / norm)
            memory[j] = momentum * memory[j] - lr * g.item()
            new_norm = norm + memory[j]
            p.data.mul_(new_norm / norm)
        final_gq = gq.item()
    print("GQ = %.2f" % (final_gq))


class MetaInitInitializer(ConstantWidthNNInitializer):
    def __init__(self, num_classes: int, image_shape: Tuple[int, int, int], initial_init: ConstantWidthNNInitializer):
        self._num_classes = num_classes
        self._image_shape = image_shape
        self._initial_init = initial_init

    def initialize(self, model: ConstantWidthNN) -> None:
        for _ in range(5):  # sometimes it degenerates and throws exception
            try:
                self._initial_init.initialize(model)
                metainit(
                    model=model,
                    criterion=nn.CrossEntropyLoss(),
                    y_size=self._num_classes,
                    x_size=(32,) + self._image_shape,  # the same parameters as for CIFAR10 experiments in the paper
                )
            except:
                continue
            break


class GaussianSubmatrixInitializer(ConstantWidthNNInitializer):
    def _initialize(self, model: ConstantWidthNN) -> None:
        if not isinstance(model.input_layer, nn.Linear):
            raise RuntimeError("GSM initialization can be used only with Fully-Connected networks")
        assert model.input_layer.out_features % 2 != 1, "the number of hidden features should be even"
        nn.init.normal_(model.input_layer.weight, std=(1/_get_input_width(model.input_layer))**0.5)
        y, x = model.input_layer.weight.shape
        y, x = y//2, x//2
        model.input_layer.weight[y:, x:] = model.input_layer.weight[:y, :x]
        model.input_layer.weight[y:, :x] = -model.input_layer.weight[:y, :x]
        model.input_layer.weight[:y, x:] = -model.input_layer.weight[:y, :x]
        if model.input_layer.bias is not None:
            nn.init.zeros_(model.input_layer.bias)
        for child in model.layers.modules():
            if isinstance(child, nn.Linear):
                nn.init.normal_(child.weight, std=(2/_get_input_width(child))**0.5)
                y, x = child.weight.shape
                y, x = y // 2, x // 2
                child.weight[y:, x:] = child.weight[:y, :x]
                child.weight[y:, :x] = -child.weight[:y, :x]
                child.weight[:y, x:] = -child.weight[:y, :x]
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
            elif isinstance(child, nn.Conv2d):
                raise RuntimeError("GSM initialization can be used only with Fully-Connected networks")

    def initialize(self, model: ConstantWidthNN) -> None:
        with torch.no_grad():
            self._initialize(model)
