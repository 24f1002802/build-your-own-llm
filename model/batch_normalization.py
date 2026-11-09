import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        pass
        # Convert inputs into NumPy arrays
        X = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)
        beta = np.array(beta, dtype=float)
        running_mean = np.array(running_mean, dtype=float)
        running_var = np.array(running_var, dtype=float)

        if training:
            batch_mean = np.mean(X, axis=0)
            batch_var = np.var(X, axis=0)
            normalized = ((X - batch_mean) / np.sqrt(batch_var + eps))
            output = gamma * normalized + beta

            # Update running statistics
            running_mean = ((1 - momentum) * running_mean + momentum * batch_mean)
            running_var = ((1 - momentum) * running_var + momentum * batch_var)
        else:
            # Inference: Use stored running statistics
            normalized = ((X - running_mean) / np.sqrt(running_var + eps))
            output = gamma * normalized + beta

        # Step 6: Round results
        return (
            np.round(output, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )
