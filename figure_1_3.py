import argparse
import itertools
import uuid
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torchvision
from typing import List, Dict, Callable

from constant_width_nn import ConstantWidthNN
from constant_width_nn_initializer import ConstantWidthNNInitializer
from train_utils import train_model, DATASETS, Dataset, Params, INITIALIZERS, NORMALIZATION_LAYERS
from figure_2_supplement_figures_plot import plot, ExperimentJson


def run_experiment(
        dataset: Dataset,
        depth: int,
        width: int,
        initializer: ConstantWidthNNInitializer,
        normalization_factory: Callable[[int], nn.Module],
        conv,
        validate_at: np.ndarray,
        use_bias: bool = True,
        learning_rate: float = 1e-4,
        batch_size: int = 90,
        use_cuda: bool = False,
        use_augmentations: bool = False,
) -> np.ndarray:
    mean, std = dataset.mean, dataset.std

    augmentations = []
    if use_augmentations:
        augmentations += [torchvision.transforms.RandomHorizontalFlip()]
    preprocessing = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=mean, std=std)]

    train_transform = torchvision.transforms.Compose(augmentations + preprocessing)
    test_transform = torchvision.transforms.Compose(preprocessing)

    train_dataset = dataset.get_training_set(train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = dataset.get_test_set(test_transform)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ConstantWidthNN(
        depth=depth,
        input_shape=dataset.input_shape,
        width=width,
        out_width=dataset.output_width,
        bias=use_bias,
        conv=conv,
        normalization_factory=normalization_factory,
    )

    initializer.initialize(model)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    test_accuracies, test_losses = train_model(train_data_loader=train_data_loader,
                                               test_data_loader=test_data_loader,
                                               criterion=criterion,
                                               optimizer=optimizer,
                                               model=model,
                                               validate_at=validate_at,
                                               use_cuda=use_cuda,
                                               )
    return test_accuracies


def main(
        max_depth: int,
        depth_step: int,
        depth_start: int,
        network_widths: List[int],
        num_repeats: int,
        dataset: Dataset,
        num_gradient_steps: int,
        initializers: List[str],
        normalization_layers: List[str],
        conv: bool,
        learning_rate: float,
) -> None:
    depths = list(range(depth_start, max_depth + 1, depth_step))

    all_params = list(map(
        lambda params: Params(width=params[0], init_method=params[1], depth=params[2], normalization_layer=params[3]),
        itertools.product(network_widths, initializers, depths, normalization_layers)),
    )
    params_to_test_accuracies: Dict[Params, List[float]] = {}
    for params in tqdm(all_params):
        params_to_test_accuracies[params] = []
        for _ in range(num_repeats):
            test_accuracy = run_experiment(
                dataset=dataset,
                depth=params.depth,
                width=params.width,
                initializer=INITIALIZERS[params.init_method],
                normalization_factory=NORMALIZATION_LAYERS[params.normalization_layer],
                conv=conv,
                validate_at=np.array([num_gradient_steps]),
                learning_rate=learning_rate,
            )[0]
            params_to_test_accuracies[params].append(test_accuracy)

    experiment_id = str(uuid.uuid4())
    experiment_json = ExperimentJson(
        depths=[params.depth for params in all_params],
        widths=[params.width for params in all_params],
        init_methods=[params.init_method for params in all_params],
        normalization_layers=[params.normalization_layer for params in all_params],
        test_accuracies=[params_to_test_accuracies[params] for params in all_params],
        dataset=dataset.name,
        num_gradient_steps=num_gradient_steps,
        num_repeats=num_repeats,
        learning_rate=learning_rate,
    )
    with open(f"{experiment_id}.json", 'w') as file:
        json.dump(experiment_json, file, indent=4)

    plot(experiment_json=experiment_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--depth-step", type=int, required=True)
    parser.add_argument("--depth-start", type=int, default=0)
    parser.add_argument("--network-widths", type=int, required=True, nargs="+")
    parser.add_argument("--num-repeats", type=int, required=True)
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--num-gradient-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--initializers", choices=INITIALIZERS.keys(), required=True, nargs="+")
    parser.add_argument("--normalization-layers", choices=NORMALIZATION_LAYERS.keys(), required=True, nargs="+")
    parser.add_argument("--conv", action='store_true', default=False)
    args = parser.parse_args()
    main(
        max_depth=args.max_depth,
        depth_step=args.depth_step,
        depth_start=args.depth_start,
        network_widths=args.network_widths,
        dataset=DATASETS[args.dataset],
        num_gradient_steps=args.num_gradient_steps,
        initializers=args.initializers,
        num_repeats=args.num_repeats,
        normalization_layers=args.normalization_layers,
        conv=args.conv,
        learning_rate=args.learning_rate,
    )

