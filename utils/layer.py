import numpy as np
from neuron import Neuron


class Layer:
    def __init__(self, neuron_count: int):
        self.neurons = [Neuron(i) for i in range(neuron_count)]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Feed the input through all neurons in the layer."""
        return np.array([neuron.get_state(inputs) for neuron in self.neurons])


if __name__ == "__main__":
    layer_1 = Layer(124)
    layer_1.forward(np.random.randn(3, 2))
    print(layer_1)
