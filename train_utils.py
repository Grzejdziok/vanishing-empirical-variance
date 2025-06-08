from dataclasses import dataclass
from typing import Callable, Dict, Union, Tuple

import tqdm

import torch
import torchvision
import torch.nn as nn
import numpy as np

from constant_width_nn_initializer import ConstantWidthNNInitializer, HeInitializer, OrthogonalInitializer, \
    ZerOInitializer, IdentityInitializer, MinimalDependenceInitializer, GlorotInitializer, LeCunInitializer, \
    DeltaOrthogonalInitializer, MetaInitInitializer, GaussianSubmatrixInitializer
from random_initialization import BernoulliZeroMeanInitializer, NormalZeroMeanInitializer, UniformZeroMeanInitializer


INITIALIZERS: Dict[str, ConstantWidthNNInitializer] = {
    "he_bernoulli": HeInitializer(BernoulliZeroMeanInitializer()),
    "he_normal": HeInitializer(NormalZeroMeanInitializer()),
    "he_uniform": HeInitializer(UniformZeroMeanInitializer()),
    "glorot_normal": GlorotInitializer(NormalZeroMeanInitializer()),
    "lecun_normal": LeCunInitializer(NormalZeroMeanInitializer()),
    "ZerO": ZerOInitializer(first_layer_random=False),
    "ZerO_star": ZerOInitializer(first_layer_random=True),
    "orthogonal": OrthogonalInitializer(),
    "identity": IdentityInitializer(),
    "minimal_dependence": MinimalDependenceInitializer(),
    "delta_orthogonal": DeltaOrthogonalInitializer(),
    "gsm": GaussianSubmatrixInitializer(),
    "metainit_cifar10_from_he_normal": MetaInitInitializer(image_shape=(3, 32, 32), num_classes=10, initial_init=HeInitializer(NormalZeroMeanInitializer())),
    "metainit_mnist_from_he_normal": MetaInitInitializer(image_shape=(3, 28, 28), num_classes=10, initial_init=HeInitializer(NormalZeroMeanInitializer()))
}

INIT_METHODS_LABELS: Dict[str, str] = {
    "he_bernoulli": "He (Bernoulli dist.)",
    "he_normal": "He (normal dist.)",
    "he_uniform": "He (uniform dist.)",
    "glorot_normal": "Glorot (normal dist.)",
    "lecun_normal": "LeCun (normal dist.)",
    "minimal_dependence": "minimal dependence (ours)",
    "delta_orthogonal": "delta-orthogonal",
    "metainit_cifar10_from_he_normal": "metainit",
    "metainit_mnist_from_he_normal": "metainit",
    "gsm": "gsm",
    "ZerO": "ZerO",
    "ZerO_star": "ZerO*",
    "orthogonal": "orthogonal",
}

INITIALIZERS_TO_KURTOSIS: Dict[str, float] = {
    "he_bernoulli": 1.0,
    "he_uniform": 1.8,
    "he_normal": 3.0,
}


NORMALIZATION_LAYERS: Dict[str, Callable[[int], nn.Module]] = {
    "no_norm": lambda _: nn.Identity(),
    "layer_norm": nn.LayerNorm,
    "batch_norm": nn.BatchNorm1d,
}


@dataclass(frozen=True)
class Params:
    depth: int
    width: int
    init_method: str
    normalization_layer: str = "no_norm"


@dataclass
class Dataset:
    get_training_set: Callable[[torchvision.transforms.Compose], torch.utils.data.Dataset]
    get_test_set: Callable[[torchvision.transforms.Compose], torch.utils.data.Dataset]
    mean: Union[float, Tuple[float, float, float]]
    std: Union[float, Tuple[float, float, float]]
    input_shape: Tuple[int, ...]
    output_width: int
    name: str
    he_random_vector_kurtosis: float
    he_random_vector_covariance_of_squares: float


MNIST = Dataset(
    get_training_set=lambda transform: torchvision.datasets.MNIST(root="./.mnist", transform=transform, download=True, train=True),
    get_test_set=lambda transform: torchvision.datasets.MNIST(root="./.mnist", transform=transform, download=True, train=False),
    mean=0.1307,
    std=0.3081,
    input_shape=(1, 28, 28),
    output_width=10,
    name="MNIST",
    he_random_vector_kurtosis=3.2843307032679196,  # estimated using 10^7 samples
    he_random_vector_covariance_of_squares=0.09726392733993318,  # estimated using 10^7 samples
)

CIFAR10 = Dataset(
    get_training_set=lambda transform: torchvision.datasets.CIFAR10(root="./.cifar10", transform=transform, download=True, train=True),
    get_test_set=lambda transform: torchvision.datasets.CIFAR10(root="./.cifar10", transform=transform, download=True, train=False),
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
    input_shape=(3, 32, 32),
    output_width=10,
    name="CIFAR10",
    he_random_vector_kurtosis=3.953255619581223,  # estimated using 10^7 samples
    he_random_vector_covariance_of_squares=0.7099125611126186,  # estimated using 10^7 samples
)

DATASETS: Dict[str, Dataset] = {
    MNIST.name: MNIST,
    CIFAR10.name: CIFAR10,
}


def validate_model(data_loader: torch.utils.data.DataLoader,
                   model: nn.Module,
                   use_cuda: bool,
                   criterion: nn.Module,) -> Tuple[float, float]:
    num_hits = 0.
    running_loss = 0.
    dataset_count = 0
    for batch_input, batch_target in data_loader:
        dataset_count += batch_input.shape[0]
        if use_cuda:
            batch_input = batch_input.cuda()
            batch_target = batch_target.cuda()
        with torch.no_grad():
            batch_output_prob = model(batch_input)
            batch_output = torch.argmax(batch_output_prob, dim=1)
            num_hits += (batch_output == batch_target).sum().float().item()
            loss = criterion(batch_output_prob, batch_target)
        running_loss += loss.item() * batch_input.size(0)
    accuracy = num_hits / dataset_count
    loss = running_loss / dataset_count
    return accuracy, loss


def train_model(train_data_loader: torch.utils.data.DataLoader,
                test_data_loader: torch.utils.data.DataLoader,
                use_cuda: bool,
                model: nn.Module,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                validate_at: np.ndarray,
                ) -> Tuple[np.ndarray, np.ndarray]:
    iterations_so_far = 0
    iter_train_data_loader = iter(train_data_loader)
    total_iterations = max(validate_at)
    test_accuracies = []
    test_losses = []
    progress_bar = tqdm.tqdm(total=total_iterations)
    if use_cuda:
        model = model.cuda()

    if 0 in validate_at:
        test_accuracy, test_loss = validate_model(test_data_loader, model, use_cuda, criterion)
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)

    while iterations_so_far < total_iterations:
        try:
            batch_input, batch_target = next(iter_train_data_loader)
            if use_cuda:
                batch_input = batch_input.cuda()
                batch_target = batch_target.cuda()
        except StopIteration:
            iter_train_data_loader = iter(train_data_loader)
            continue
        model.train()
        batch_output_prob = model(batch_input)
        loss = criterion(batch_output_prob, batch_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        iterations_so_far += 1
        progress_bar.update()

        if iterations_so_far in validate_at:
            # Find accuracy and loss
            model.eval()
            test_accuracy, test_loss = validate_model(test_data_loader, model, use_cuda, criterion)
            test_accuracies.append(test_accuracy)
            test_losses.append(test_loss)
    progress_bar.close()
    test_accuracies = np.asarray(test_accuracies)
    test_losses = np.asarray(test_losses)
    return test_accuracies, test_losses
