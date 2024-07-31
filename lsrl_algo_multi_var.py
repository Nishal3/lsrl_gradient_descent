from random import random
from typing import Union, Sequence

Num = Union[int, float]


def residual_sum_of_squares(data: Sequence[Sequence[Num]], vector_w: Sequence[Num], b: Num) -> Num:
    """
    Returns the residual sum of squares (RSS) with the given data, weights, and intercept.

    Variables:
    data:       A tuple with tuples inside representing points on an n dimensional grid
    vector_w:   A tuple of weights corresponding to a feature of a point
    b:          A number which represents the intercept

    >>> _data = ((0.022, 0.12, 0.01), (0.859, 4.963, 1.548), (13.324, 2.714, 19.352),\
                 (17.454, 26.582, 1.066), (10.907, 1.249, 34.12), (22.627, 36.543, 18.503),\
                 (26.145, 9.041, 34.738), (26.848, 36.133, 37.874), (31.201, 18.57, 33.332),\
                 (73.594, 44.598, 81.683), (82.6, 3.817, 91.421))
    >>> _vector_w, _b = (0, 0), 0
    >>> residual_sum_of_squares(_data, _vector_w, _b)
    20666.670547
    :param data:
    :param vector_w:
    :param b:
    :return:
    """
    rss = 0
    for point in data:
        rss += point_square_error(point, vector_w, b)

    return rss

def point_square_error(point: Sequence[Num], vector_w: Sequence[Num], b: Num) -> Num:
    """
    Calculates the square error at a given point.

    Variables:
    data:       A tuple with tuples inside representing points on an n dimensional grid
    vector_w:   A tuple of weights corresponding to a feature of a point
    b:          A number which represents the intercept

    >>> _point = (0.4, 0.681, 0.237)
    >>> _vector_w, _b = (0, 0), 0
    >>> point_square_error(_point, _vector_w, _b)
    0.056169

    :param point:
    :param vector_w:
    :param b:
    :return:
    """

    # Error always has a +b at the end
    error = b

    for index, feature in enumerate(point[:-1]):
        error += vector_w[index] * feature

    error = (error - point[-1]) ** 2

    return error

def gradient_descent_input_validation(data: Sequence[Sequence[Num]], len_vector_w: int) -> bool:
    """
    Checks whether `data` has enough variables inside each individual data point to match with the slopes.

    Returns a boolean value to indicate whether it is valid or not.

    >>> _data = ((0.022, 0.12, 0.01), (0.859, 4.963, 1.548), (13.324, 2.714, 19.352),\
                 (17.454, 26.582, 1.066), (10.907, 1.249, 34.12), (22.627, 36.543, 18.503),\
                 (26.145, 9.041, 34.738), (26.848, 36.133, 37.874), (31.201, 18.57, 33.332),\
                 (73.594, 44.598, 81.683), (82.6, 3.817, 91.421))
    >>> _len_vector_w = 2
    >>> gradient_descent_input_validation(_data, _len_vector_w)
    True

    :param data:
    :param len_vector_w:
    :return:
    """
    if not data:
        return False

    for point in data:
        if len(point) != len_vector_w + 1:
            return False

    return True


def gradient_descent(data: Sequence[Sequence[Num]], vector_w: Sequence[Num], b: Num, alpha=0.001) -> (Num, Sequence[Num]):
    """
    Ran repeatedly, this function minimizes the mean squared error(MSE) cost function for linear regression for N Variables

    You can input as many dimensions of data as you want as long as vector_w = len(data[0]) - 1.

    Variables:
    data:           Array of tuples representing points. The length of each tuple in the list needs to be the same
    vector_w:       Array of weights
    b:              y-intercept, when all x_vectors(input variables) are 0, y(output variable) is predicted to be b
    alpha:          The learning rate, default is 0.001


    >>> _data = (\
            (0.4, 0.681, 0.237),\
            (2.36, 5.634, 8.177),\
            (6.413, 2.104, 3.126),\
            (17.454, 26.582, 1.066),\
            (10.907, 1.249, 34.12),\
            (22.627, 36.543, 18.503),\
            (26.145, 9.041, 34.738),\
            (26.848, 36.133, 37.874),\
            (31.201, 18.57, 33.332),\
            (73.594, 44.598, 81.683),\
            (82.6, 3.817, 91.421)\
       )
    >>>
    >>> _w = (0, 0)
    >>> _b = 0
    >>> gradient_descent(_data, _w, _b)
    (0.03129790909090909, [1.5796973580909088, 0.6448463882727273])

    :param data:
    :param vector_w:
    :param b:
    :param alpha:
    :return:
    """

    dim = len(data[0])
    len_data = len(data)
    # Handling b separately
    b_gradient = 0
    gradients = [0] * (dim - 1)

    # Tabulated derivatives with no point coefficients
    tabulated_derivatives = [0] * len_data

    valid_parameters = gradient_descent_input_validation(data=data, len_vector_w=len(vector_w))

    if not valid_parameters:
        raise ValueError(f"Data length mismatch. Expected every point in `data` to be of length {len(vector_w) + 1}")

    # Calculating the gradient for 'b', it is the only one that doesn't get multiplied by an element of a point
    for index, point in enumerate(data):
        cur_derivative = derivative_of_point(point, vector_w, b, len_data)
        b_gradient += cur_derivative
        tabulated_derivatives[index] = cur_derivative

    # Updating term b
    b_gradient = b - alpha * b_gradient

    # Utilizing tabulation to increase efficiency
    cur_weight_index = 0
    while cur_weight_index < dim - 1:
        for index, point in enumerate(data):
            gradients[cur_weight_index] += tabulated_derivatives[index] * point[cur_weight_index]

        # Updating the weights
        gradients[cur_weight_index] = vector_w[cur_weight_index] - alpha * gradients[cur_weight_index]
        cur_weight_index += 1

    return b_gradient, gradients


def derivative_of_point(point: Sequence[Num], vector_w: Sequence[Num], b: Num, len_data: int, deriv_index=-1) -> float:
    """
    Gets the derivative value of the mean squared error for a specific point.

    Variables:
    data:           Array of tuples representing points. The length of each tuple in the list needs to be the same
    vector_w:       Array of weights
    b:              y-intercept, when all x_vectors(input variables) are 0, y(output variable) is predicted to be b
    len_data:       Number of points present in data
    deriv_index:    The index of a weight which you want to take the derivative with respect to

    >>> _point = (0.4, 0.681, 0.237)
    >>> _vector_w, _b = (1, 1), 0
    >>> _len_data = 11
    >>> _deriv_index = 0
    >>> derivative_of_point(_point, _vector_w, _b, _len_data, _deriv_index)
    0.03069090909090909


    :param point:
    :param vector_w:
    :param b:
    :param len_data:
    :param deriv_index:
    :return:
    """

    deriv = dot_product(point[:-1], vector_w)

    # Derivative of mean squared error at the given deriv_index. Associates with a coefficient of a value in vector_w
    if deriv_index >= 0:
        deriv = (deriv + b - point[-1]) * point[deriv_index] / len_data
    else:
        deriv = (deriv + b - point[-1]) / len_data

    return deriv


def dot_product(vector_1: Sequence[Num], vector_2: Sequence[Num]) -> Num:
    """
    Returns dot product of two vectors.

    >>> _vector1 = [1, 2, 3, 4]
    >>> _vector2 = [2, 6, 1, 9]
    >>> dot_product(_vector1, _vector2)
    53

    :param vector_1:
    :param vector_2:
    :return:
    """
    res = 0

    for x, y in zip(vector_1, vector_2, strict=True):
        res += x * y

    return res


if __name__ == '__main__':
    x, y, z = [round(random() * i, 3) for i in range(1, 102, 10)],\
              [round(random() * i, 3) for i in range(1, 102, 10)],\
              [round(random() * i, 3) for i in range(1, 102, 10)]

    data = list(zip(x, y, z, strict=True))

    data = (
        (0.4, 0.681, 0.237),
        (2.36, 5.634, 8.177),
        (6.413, 2.104, 3.126),
        (17.454, 26.582, 1.066),
        (10.907, 1.249, 34.12),
        (22.627, 36.543, 18.503),
        (26.145, 9.041, 34.738),
        (26.848, 36.133, 37.874),
        (31.201, 18.57, 33.332),
        (73.594, 44.598, 81.683),
        (82.6, 3.817, 91.421)
    )
    slopes = (0, 0)
    b = 0

    epochs = 100000

    # Still working on the function
    error_deriv = 0

    for i in range(epochs):
        print(f"RSS: {residual_sum_of_squares(data, slopes, b)}", end="\t")
        b, slopes = gradient_descent(data, slopes, b)
        print(f"Vector W: {slopes}\tb: {b}")

    print(data)

