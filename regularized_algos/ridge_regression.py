from random import randint
from typing import Sequence, Union

Num = Union[int, float]


def residual_sum_of_squares(data: Sequence[Sequence[Num]], vector_w: Sequence[Num], b: Num, alpha: Sequence[Num]):
    error = 0

    for point in data:
        pass


def point_square_error(point: Sequence[Num], vector_w: Sequence[Num], b: Num, alpha: Sequence[Num]):
    point_error = 0

    for feature, weight, penalty in zip(point[:-1], vector_w, alpha):
        point_error += (feature * weight) ** 2 + penalty * (weight ** 2)

    point_error = point_error - point[-1]

    return point_error
