from random import random
from typing import Tuple


def gradient_descent(data: Tuple[Tuple[float]], vector_w: Tuple[float], b: float) -> Tuple:
    """
    Ran repeatedly, this function minimizes the mean squared error(MSE) cost function for linear regression for N Variables

    You can input as many dimensions of data as you want as long as vector_w = len(data[0]) - 1.

    Variables:
    data:       Array of tuples representing points. The length of each tuple in the list needs to be the same
    vector_w:   Array of weights
    b:          Intercept, when all x_vectors(input variables) are 0, y(output variable) is predicted to be b


    >>> _data = (
    >>> (0.022, 0.12, 0.01), (0.859, 4.963, 1.548), (13.324, 2.714, 19.352),
    >>> (17.454, 26.582, 1.066), (10.907, 1.249, 34.12), (22.627, 36.543, 18.503),
    >>> (26.145, 9.041, 34.738), (26.848, 36.133, 37.874), (31.201, 18.57, 33.332),
    >>> (73.594, 44.598, 81.683), (82.6, 3.817, 91.421)
    >>> )
    >>>
    >>> _w = (0, 0)
    >>> _b = 0
    >>> gradient_descent(_data, _w, _b)
    (-32.14972727272727, -1599.6733982727274, -645.5188912727273)

    :param data:
    :param vector_w:
    :return:
    """
    dim = len(data[0])
    len_data = len(data)
    gradients = [b] * dim  # This is the 'b' value. Every sum includes this, so we start everything with it

    # Calculating the gradient for 'b', it is the only one that doesn't get multiplied by an element of a point
    for point in data:
        for index, val in enumerate(point):  # point = (3, 2, 2, 9), first index=0, val=3. then index=1,
            if index == len_data - 1:        # val=2, etc. The purpose is to match index values to slopes
                continue
            gradients[0] += vector_w[index] * val

        gradients[0] = (gradients[0] - point[-1])

    for i in range(1, dim):
        break
        #gradients[i] = sum([val * slope for val, slope in zip()]) / len_data


def derivative_of_point(point: Tuple[float], vector_m: Tuple[float], b: float, len_data: int, deriv_index=None) -> float:
    """
    Gets the derivative value of the mean squared error for a specific point.

    :param point:
    :param vector_m:
    :param b:
    :param len_data:
    :param deriv_index:
    :return:
    """

    deriv = 0

    for point_val, slope in zip(point[:-1], vector_m, strict=True):
        deriv += slope * point

    # Derivative of mean squared error at the given deriv_index. Associates with a coefficient of a value in vector_m
    if deriv_index:
        deriv = (deriv + b - point_val[-1]) * point[deriv_index] / len_data
    else:
        deriv = (deriv + b - point_val[-1]) / len_data

    return deriv


if __name__ == '__main__':
    x, y, z = [round(random() * i, 3) for i in range(1, 102, 10)],\
              [round(random() * i, 3) for i in range(1, 102, 10)],\
              [round(random() * i, 3) for i in range(1, 102, 10)]

    data = list(zip(x, y, z, strict=True))

    data = [
        (0.4, 0.681, 0.237),
        (2.36, 5.634, 8.177),
        (6.413, 2.104, 3.126),
        (17.419, 22.034, 29.193),
        (22.78, 13.524, 6.192),
        (31.03, 36.753, 21.017),
        (26.6, 40.423, 44.091),
        (59.431, 54.743, 20.141),
        (52.478, 14.786, 38.24),
        (33.527, 13.546, 50.097),
        (34.375, 93.439, 83.492)
    ]
    """data = [
        (0.022, 0.12, 0.01), (0.859, 4.963, 1.548), (13.324, 2.714, 19.352),
        (17.454, 26.582, 1.066), (10.907, 1.249, 34.12), (22.627, 36.543, 18.503),
        (26.145, 9.041, 34.738), (26.848, 36.133, 37.874), (31.201, 18.57, 33.332),
        (73.594, 44.598, 81.683), (82.6, 3.817, 91.421)]
"""
    epochs = 1000

    # Still working on the function


