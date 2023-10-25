import numpy as np
import typing


class Neuron:

    '''
    weights: 3x2 weight matrix
    state: 1x3 state input
    Output: delta input
    '''

    def __init__(self, input: float, mu: float = 1, sigma: float = 1,  bias: float = 0.01):
        '''
        Creates a random neuron
        '''

        self.weights = [[np.random.normal(mu, sigma)
                         for _ in range(2)] for _ in range(3)]
        # -1 adds direction to magnituide of vector state
        self.state = [input, np.random.normal(mu, sigma), 1]

        # print(self.weights)
        # print(self.state)

    def feed_forward(self, parent_node):
        '''
        Feedforward in neural net uses the tanh activation function
        to bound answer within -1-1 range, introduces non-linearity
        '''

    # Activation Functions
    @staticmethod
    def activate(x: np.ndarray) -> np.ndarray:
        """The activation function (tanh)."""
        return np.tanh(x)

    @staticmethod
    def activate_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of the activation function."""
        return 1.0 - np.tanh(x) ** 2

    # Getters & Setters

    def get_state(self) -> list:
        '''
        show and return current state
        '''
        print(self.state)
        return self.state

    def set_state(self, vector: list):
        '''
        update state
        '''
        self.state = vector

    def get_weights(self) -> list:
        '''
        print & show weights
        '''
        print(self.weights)
        return self.weights

    def set_weights(self, vector: list):
        '''
        set weights to vector
        '''
        self.weights = vector


if __name__ == "__main__":
    n1 = Neuron(1.089)
