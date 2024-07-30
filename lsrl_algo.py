from random import random


def mean_squared_error_2d(data, m, b):
    error = 0
    for x, y in data:
        error += (m * x + b - y) ** 2

    return error


def gradient_descent_2d(data, m, b, lr=0.0001):
    b_gradient = 0  # Derivitave of MSE formula w/ respect to b: 1 / len(data) * (sum(x_1 * m_1 + b - y))
    m_gradient = 0  # Derivative of MSE formula w/ respect to m: 1 / len(data) * (sum(x_1 * m_1 + b - y) * x_1)

    len_data = len(data)

    for x, y in data:
        b_gradient += (x * m + b - y) / len_data
        m_gradient += ((x * m + b - y) * x) / len_data

    # Simultaneous assignment to vars m and b
    m, b = m - lr * m_gradient, b - lr * b_gradient
    return m, b


def mean_squared_error_3d(data, m_1, m_2, b):
    error = 0
    for x, y, z in data:
        error = ((x * m_1 + y * m_2 + b) - z) ** 2  # MSE with more variables is not that bad, pretty intuitive

    return error


def gradient_descent_3d(data, m_1, m_2, b, lr=0.00005):
    b_gradient = 0
    m_1_gradient = 0
    m_2_gradient = 0

    len_data = len(data)

    for x, y, z in data:
        b_gradient += (x * m_1 + y * m_2 + b - z) / len_data
        m_1_gradient += ((x * m_1 + y * m_2 + b - z) * x) / len_data
        m_2_gradient += ((x * m_1 + y * m_2 + b - z) * y) / len_data

    m_1, m_2, b = m_1 - lr * m_1_gradient, m_2 - lr * m_2_gradient, b - lr * b_gradient
    return m_1, m_2, b


if __name__ == '__main__':
    # Randomly generated test cases:
    x, y, z = [round(random() * i, 3) for i in range(1, 102, 10)],\
              [round(random() * i, 3) for i in range(1, 102, 10)],\
              [round(random() * i, 3) for i in range(1, 102, 10)]

    # Changing from the lazy zip iterator to a normal list
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

    epochs = 1

    m_1, m_2, b = 0, 0, 0
    # Loop to see how the variables and MSE get minimized
    for i in range(epochs):
        print(f"MSE:\t{mean_squared_error_3d(data=data, m_1=m_1, m_2=m_2, b=b)}", end="\t")
        m_1, m_2, b = gradient_descent_3d(data=data, m_1=m_1, m_2=m_2, b=b)
        print(f"m1:\t{m_1}\tm2:\t{m_2}\tb:\t{b}")

    # Just in case there are some errors with specific data
    print(data)
