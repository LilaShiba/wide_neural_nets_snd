import numpy as np
import typing
from neuron import Neuron


class Layer:
    '''
    a single layer of X neurons
    complex representation
    '''

    def __init__(self, name: str, layer: int, neurons: list = None):
        '''
        Creates a randomized neuron with no current state
        '''
        self.name = name
        self.layer = layer
        self.neurons = neurons
        if not neurons:
            self.create_neurons()

    def create_neurons(self):
        '''
        Create a layer of randomized neurons as 2x2 matrix
        '''
        self.neurons = [Neuron(self.name, i) for i in range(self.layer)]
        return self.neurons

    def update(self, input_matrix: list[list]):
        '''
        feed-forward updating of neurons in layer
        - Other: Neuron
        '''
        # project
        for neuron in self.neurons:
            for training_example in input_matrix:
                neuron.update(training_example)

    def set_params(self, params):
        '''
        Set the neural parameters matrix
        '''
        self.neurons = params

    def get_params(self):
        '''
        Get the neural parameters matrix
        '''
        print(self.neurons)
        return self.neurons


if __name__ == "__main__":
    n1 = Neuron('1', 1)
    n2 = Neuron('2', 2)
    n1.update(n2)
    n1.get_params()
