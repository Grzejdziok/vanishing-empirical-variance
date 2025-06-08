from typing import Tuple
import numpy as np


def get_next_kurtosis(
        current_c: float,
        current_kappa_x: float,
        negative_slope: float,
        width: int,
        weight_init_kurtosis: float,
) -> Tuple[float, float]:

    a = negative_slope
    w = width
    kappa_w = weight_init_kurtosis

    x = np.array([
        [current_kappa_x],
        [current_c],
    ])

    A = np.array([
        [2*(a**4+1)*kappa_w/w/(a**2+1)**2, 3*(w-1)/w],
        [2*(a**4+1)/w/(a**2+1)**2, (w-1)/w],
    ])

    b = np.array([
        [3*(w-1)/w],
        [-1/w],
    ])

    kappa_x, c = np.dot(A, x) + b

    return float(c), float(kappa_x)


def calculate_theoretical_kurtosis(
        layer_depth: int,
        negative_slope: float,
        width: int,
        weight_init_kurtosis: float,
        input_init_kurtosis: float,
        covariance_of_squares: float = 0.,
) -> float:
    c = covariance_of_squares
    kappa_x = input_init_kurtosis
    for i in range(layer_depth):
        c, kappa_x = get_next_kurtosis(
            current_c=c,
            current_kappa_x=kappa_x,
            negative_slope=negative_slope,
            width=width,
            weight_init_kurtosis=weight_init_kurtosis,
        )
    return kappa_x
