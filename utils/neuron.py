import numpy as np
import typing


class Neuron:
    '''
    a single neuron for feedforward nn
    complex representation
    '''

    def __init__(self, name: str, layer: int):
        '''
        Creates a randomized neuron with no current state
        '''
        self.name = name
        self.layer = layer
        # input,  weight, current_state
        # output, bias, new_state
        self.neuron = np.random.randn(2, 2)
        self.state = None
        self.connections = {}

    def train(self, parent: object):
        '''
        feed-forward updating of neuron from parent
        if input layer, input is parent
        - Other: Neuron
        '''
        # project
        self.neuron = np.dot(self.neuron, parent.neuron)
        # activate
        self.state = np.tanh(self.neuron)
        return self.state

    def set_params(self, params):
        '''
        Set the neural parameters matrix
        '''
        self.neuron = params

    def get_params(self):
        '''
        Get the neural parameters matrix
        '''
        print(self.neuron)
        return self.neuron


if __name__ == "__main__":
    n1 = Neuron('1', 1)
    n2 = Neuron('2', 2)
    n1.update(n2)
    n1.get_params()
