import numpy as np
from utils.neuron import Neuron
import matplotlib.pyplot as plt


class Layer:
    def __init__(self, neuron_count: int):
        self.neurons = [Neuron(np.random.uniform(-1, 1))
                        for _ in range(neuron_count)]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Feed the input through all neurons in the layer.
        inputs is neuron from previous layer :) 

        """
        self.neurons = np.array([neuron.feed_forward(inputs)
                                 for neuron in self.neurons])


if __name__ == "__main__":
    transducer = Neuron(1.78293)
    layer_1 = Layer(12)
    layer_1.forward(transducer)
    print(layer_1)
    for neuron in layer_1.neurons:
        plt.plot(neuron.signal)
