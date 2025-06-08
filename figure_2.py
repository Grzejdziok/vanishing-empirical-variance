from typing import List, Dict
import argparse
from tqdm import tqdm
import json

import torch
import torch.utils.data
import torchvision
import matplotlib

from constant_width_nn import ConstantWidthNN
from output_statistics import OutputStatisticsMeasurement
from train_utils import DATASETS, Dataset, INITIALIZERS
from figure_1_plot import ExperimentJson, plot


def main(
        depth: int,
        width: int,
        num_repeats: int,
        dataset: Dataset,
        initializers: List[str],
        conv: bool,
) -> None:
    matplotlib.rc('font', size=16)
    matplotlib.rc('image', cmap="cividis")
    mean, std = dataset.mean, dataset.std
    preprocessing = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=mean, std=std)]
    train_transform = torchvision.transforms.Compose(preprocessing)

    train_dataset = dataset.get_training_set(train_transform)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1000, num_workers=8)

    initializer_to_empirical_variances = {}
    for initializer in tqdm(initializers):
        initializer_to_empirical_variances[initializer] = []
        model = ConstantWidthNN(
            depth=depth,
            input_shape=dataset.input_shape,
            width=width,
            out_width=dataset.output_width,
            bias=True,
            conv=conv,
        )
        if torch.cuda.is_available():
            model = model.cuda()
        for _ in tqdm(range(num_repeats)):
            INITIALIZERS[initializer].initialize(model)
            with torch.no_grad():
                model.set_statistics_measurement(OutputStatisticsMeasurement())
                for inputs, _ in train_data_loader:
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                    model(inputs)
                initializer_to_empirical_variances[initializer].append(model.statistics_measurement.empirical_variances())

        with open(f"empirical_variances_{initializer}_{dataset.name}_w{width}_d{depth}_x{num_repeats}.json",
                  'w') as file:
            data = ExperimentJson(
                width=width,
                empirical_variances=initializer_to_empirical_variances[initializer],
                dataset=dataset.name,
                num_repeats=num_repeats,
                init_method=initializer,
            )
            json.dump(data, file)

        plot(
            initializer=initializer,
            empirical_variances=initializer_to_empirical_variances[initializer],
            quantiles=[0.9, 0.99, 0.999],
            dataset=dataset.name,
            width=width,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num-repeats", type=int, required=True)
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--initializers", choices=INITIALIZERS.keys(), required=True, nargs="+")
    parser.add_argument("--conv", action='store_true', default=False)
    args = parser.parse_args()
    main(
        depth=args.depth,
        width=args.width,
        dataset=DATASETS[args.dataset],
        initializers=args.initializers,
        num_repeats=args.num_repeats,
        conv=args.conv,
    )

