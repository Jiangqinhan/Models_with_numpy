import numpy as np
from Base_class import Layer
class ReLU(Layer):
    def __init__(self):
        self._last_input = None

    def forward(self, X):
        self._last_input = X
        return np.maximum(0, X)

    def backward(self, prev_grads):
        assert prev_grads.shape == self._last_input.shape

        local_grads = np.zeros_like(self._last_input)
        local_grads[self._last_input > 0] = 1.0

        return prev_grads * local_grads