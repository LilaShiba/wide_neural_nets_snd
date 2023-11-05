import numpy as np
from utils.neuron import Neuron
import matplotlib.pyplot as plt
import heapq


class Layer:
    def __init__(self, neuron_count: int):
        self.neurons = [Neuron(1.221)
                        for _ in range(neuron_count)]

    def forward(self, input: object) -> np.ndarray:
        """
        Feed the input through all neurons in the layer.
        inputs is neuron from previous layer :) 

        """
        self.neurons = np.array([neuron.feed_forward(neuron.signal)
                                 for neuron in self.neurons])

    def graph(self):
        '''
        show vectors
        input, state
        '''
        for n in self.neurons:
            t = [i for i in n.signal if i != 1]
            plt.plot(t)
        plt.title('Layer Output')
        plt.xlabel('time step')
        plt.ylabel('input value -> state')
        plt.show()


if __name__ == "__main__":
    transducer = Neuron(1.78293)
    layer_1 = Layer(12)
    layer_1.forward(transducer)
    print(layer_1)
    for neuron in layer_1.neurons:
        plt.plot(neuron.signal)
        plt.show()
