import argparse
from typing import List
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict
import json

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from kurtosis_utils import calculate_theoretical_kurtosis
from train_utils import Params, INITIALIZERS_TO_KURTOSIS, DATASETS, INIT_METHODS_LABELS


class ExperimentJson(TypedDict):
    depths: List[int]
    widths: List[int]
    init_methods: List[str]
    test_accuracies: List[List[float]]
    dataset: str
    num_gradient_steps: int
    num_repeats: int
    normalization_layers: List[str]
    learning_rate: float


def plot(experiment_json: ExperimentJson, skip_legend: bool = False) -> None:
    matplotlib.rc('font', size=14)
    matplotlib.rc('image', cmap="cividis")

    num_repeats = experiment_json["num_repeats"]
    dataset = DATASETS[experiment_json["dataset"]]
    num_gradient_steps = experiment_json["num_gradient_steps"]
    learning_rate = experiment_json["learning_rate"]

    params_list = list(map(
        lambda t: Params(width=t[0], depth=t[1], init_method=t[2], normalization_layer=t[3]),
        zip(experiment_json["widths"], experiment_json["depths"], experiment_json["init_methods"], experiment_json["normalization_layers"])),
    )
    params_to_test_accuracies = {
        params: test_accuracies for params, test_accuracies in zip(params_list, experiment_json["test_accuracies"])
    }
    unique_widths = sorted(list(set(params.width for params in params_list)))
    unique_depths = sorted(list(set(params.depth for params in params_list)))
    unique_init_methods = sorted(list(set(params.init_method for params in params_list)))
    unique_normalization_layers = sorted(list(set(params.normalization_layer for params in params_list)))
    max_depth = max(unique_depths)
    max_accuracy = max(max(v) for v in params_to_test_accuracies.values())
    max_accuracy_rounded_to_01 = (math.ceil(max_accuracy*10)+1e-5)/10

    plt.clf()
    kurtoses = []
    test_accuracies = []
    for params in params_list:
        if params.init_method not in INITIALIZERS_TO_KURTOSIS.keys():
            continue
        output_kurtosis = calculate_theoretical_kurtosis(
            input_init_kurtosis=dataset.he_random_vector_kurtosis,
            covariance_of_squares=dataset.he_random_vector_covariance_of_squares,
            layer_depth=params.depth,
            negative_slope=0.,
            width=params.width,
            weight_init_kurtosis=INITIALIZERS_TO_KURTOSIS[params.init_method],
        )
        params_test_accuracies = params_to_test_accuracies[params]
        kurtoses.extend(len(params_test_accuracies) * [output_kurtosis])
        test_accuracies.extend(params_test_accuracies)

    if len(kurtoses) > 0:
        matplotlib.rc('font', size=20)
        kurtoses = np.array(kurtoses)
        test_accuracies = np.array(test_accuracies)
        plt.scatter(kurtoses, test_accuracies)
        plt.yticks(ticks=np.arange(0.1, stop=max_accuracy_rounded_to_01, step=0.1))
        plt.xscale("log")
        plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9)
        plt.title(dataset.name)
        filename = f"kurtosis_to_test_accuracy_{dataset.name}_{len(kurtoses)}samples_d{max_depth}_w{max(unique_widths)}_lr{learning_rate}"
        plt.savefig(f"{filename}.png")
        plt.savefig(f"{filename}.eps")

        plt.clf()
    matplotlib.rc('font', size=18)
    # accuracy over depth, separately for all widths and all init methods
    init_method_order = {
        "lecun_normal": 1,
        "glorot_normal": 2,
        "orthogonal": 3,
        "he_normal": 4,
        "gsm": 5,
        "metainit_cifar10_from_he_normal": 6,
        "metainit_mnist_from_he_normal": 7,
        "identity": 8,
        "zero": 9,
        "minimal_dependence": 10,
    }
    params_iterator = sorted(list(set((params.init_method, params.width, params.normalization_layer) for params in params_list)), key=lambda a: (init_method_order.get(a[0].lower(), np.inf), a[1], a[2] != "no_norm", a[2]))
    for init_method, network_width, normalization_layer in params_iterator:
        test_accuracies_over_depths = [params_to_test_accuracies[Params(depth=depth, width=network_width, init_method=init_method, normalization_layer=normalization_layer)] for depth in unique_depths]

        average = np.mean(test_accuracies_over_depths, axis=1)
        errors = np.vstack((-np.amin(test_accuracies_over_depths, axis=1) + average, np.maximum(np.amax(test_accuracies_over_depths, axis=1) - average, 0.)))
        width_label = f"w={network_width}" if len(unique_widths) > 1 else ""
        init_method_label = INIT_METHODS_LABELS.get(init_method, init_method) if len(unique_init_methods) > 1 else ""
        normalization_label = normalization_layer if len(unique_normalization_layers) > 1 else ""
        labels = list(filter(lambda lbl: lbl != "", [width_label, init_method_label, normalization_label]))
        label = ", ".join(labels)
        plt.errorbar(unique_depths, average, yerr=errors, capsize=2, label=label)
    plt.xticks(np.linspace(0, max_depth, num=6, dtype=int))
    plt.yticks(ticks=np.arange(0.1, stop=max_accuracy_rounded_to_01, step=0.1))
    plt.ylim((0.09, max_accuracy_rounded_to_01))
    plt.title(f"w={unique_widths[0]}")
    if not skip_legend:
        plt.legend(loc=(1.04, 0))
    filename = f"test_acc_{dataset.name}_{'_'.join(unique_init_methods)}_g{num_gradient_steps}_d{'_'.join(map(str, unique_depths))}_w{'_'.join(map(str, unique_widths))}_n{'_'.join(map(str, unique_normalization_layers))}_x{num_repeats}_lr{learning_rate}"
    plt.savefig(f"{filename}.png", bbox_inches='tight')
    plt.savefig(f"{filename}.eps", bbox_inches='tight')


def read_experiment_json(filename: str) -> ExperimentJson:
    with open(filename) as file:
        data = json.load(file)
    if "normalization_layers" not in data.keys():
        data["normalization_layers"] = ["no_norm" for _ in data["widths"]]
    if "learning_rate" not in data.keys():
        data["learning_rate"] = 1e-4
    experiment_json = ExperimentJson(**data)
    return experiment_json


def main(filename: str, skip_legend: bool=False) -> None:
    experiment_json = read_experiment_json(filename)
    plot(experiment_json=experiment_json, skip_legend=skip_legend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str,)
    parser.add_argument("--skip-legend", action="store_true", default=False)
    args = parser.parse_args()
    main(
        filename=args.filename,
        skip_legend=args.skip_legend,
    )
