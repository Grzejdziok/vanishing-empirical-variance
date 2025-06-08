from typing import TypedDict, List
import argparse
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from train_utils import DATASETS, INIT_METHODS_LABELS
import scipy.stats


class ExperimentJson(TypedDict):
    width: int
    init_method: str
    empirical_variances: List[List[float]]
    dataset: str
    num_repeats: int


def plot(
        initializer: str,
        empirical_variances: List[List[float]],
        quantiles: List[float],
        dataset: str,
        width: int,
) -> None:
    dataset = DATASETS[dataset]
    plt.clf()
    matplotlib.rc('font', size=22)
    num_repeats = len(empirical_variances)
    depth = len(empirical_variances[0])-1
    for quantile in quantiles:
        quantile_over_depth = scipy.stats.mstats.mquantiles(empirical_variances, prob=[quantile], axis=0)[0]
        plt.plot(list(range(depth+1)), quantile_over_depth, label=f"p={quantile:.3f}", linewidth=1.0)
    plt.legend()
    filename = f"empirical_variance_{dataset.name}_{initializer}_d{depth}_w{width}_x{num_repeats}_q{'_'.join([f'{q:.3f}' for q in quantiles])}"
    plt.title(INIT_METHODS_LABELS[initializer])
    plt.yscale("log")
    plt.gca().set_ylim(bottom=10**-3)
    plt.xticks(np.linspace(0, 100, num=6, dtype=int))
    plt.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.88)
    plt.savefig(f"{filename}.png")
    plt.savefig(f"{filename}.eps")
    plt.close()


def main(filename: str, quantiles: List[float]) -> None:
    with open(filename) as file:
        data = ExperimentJson(**json.load(file))
    plot(
        initializer=data["init_method"],
        empirical_variances=data["empirical_variances"],
        dataset=data["dataset"],
        quantiles=quantiles,
        width=data["width"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str,)
    parser.add_argument("--quantiles", type=float, nargs="+")
    args = parser.parse_args()
    main(filename=args.filename, quantiles=args.quantiles)
