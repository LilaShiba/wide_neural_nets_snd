import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self, layer: int, bias: float = 0.01):
        self.bias = bias
        self.layer = layer
        # Making neuron a 2D matrix for weights
        self.weights = np.random.randn(3, 2)
        self.current_state = self.activate(self.weights + self.bias)

    def activate(self, x: np.ndarray) -> np.ndarray:
        """The activation function (tanh)."""
        return np.tanh(x)

    def activate_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of the activation function."""
        return 1.0 - np.tanh(x) ** 2

    def get_state(self, inputs: np.ndarray) -> np.ndarray:
        """Calculate neuron output."""
        # Multiplying the inputs with the weight matrix and adding bias
        self.weights = self.activate(
            np.cross(self.weights, inputs) + self.bias)
        self.current_state = self.weights
        return self.current_state

    def show_state(self):
        '''Graph current state in activation function'''
        print(self.current_state)
        plt.title(self.layer)
        plt.plot(self.current_state)
        plt.show()

    def set_params(self, weights: np.ndarray, bias: float) -> None:
        """Set the neuron parameters."""
        self.weights = weights
        self.bias = bias

    def get_params(self) -> tuple:
        """Get the neuron parameters."""
        return self.weights, self.bias

    @staticmethod
    def train(neurons, inputs):
        neurons[0].forward(inputs)
        for idx, n1 in enumerate(neurons):
            for idx2 in range(1, len(neurons)-1):
                n2 = neurons[idx2]
                n2.forward(n1.current_state)
            n1.show_state()


if __name__ == "__main__":

    n1 = Neuron(1)
    n2 = Neuron(2)
    n3 = Neuron(3)
    layer = [n1, n2, n3]

    # Random inputs of shape (2,)
    inputs = np.random.randn(3)

    for _ in range(3):
        Neuron.train(layer, inputs)


# TODO transduce signal to 3x2 structure
