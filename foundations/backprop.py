import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        pass

        z = np.dot(w,x) + b
        y_hat = 1 / (1 + np.exp(-z))

        error = y_hat - y_true                 # derivative of your MSE loss
        sigmoid_deriv = y_hat * (1.0 - y_hat)  # Using the sigmoid derivative

        delta = error * sigmoid_deriv

        dL_dw = np.round(delta * x, 5)
        dL_db = np.round(delta, 5)

        return (dL_dw, dL_db)





