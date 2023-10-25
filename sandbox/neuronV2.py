import numpy as np
import typing
import matplotlib.pyplot as plt


class Neuron:

    '''
    weights: 3x2 weight matrix
    state: 1x3 state input
    Output: delta input
    '''

    def __init__(self, input: float, mu: float = 1, sigma: float = 1,  bias: float = 0.01):
        '''
        Creates a random neuron
        self.weights = 3x2
        self.signal = 3x1
        '''
        self.input = input
        self.output = input
        self.bias = bias

        self.weights = np.array([[np.random.normal(mu, sigma)
                                  for _ in range(2)] for _ in range(3)])
        # -1 adds direction to magnituide of vector state
        self.signal = np.array([input, np.random.normal(mu, sigma), 1])
        # Introduce Time :)
        self.state = np.random.randint(-1, 1)

    def feed_forward(self, parent_node):
        '''
        Feedforward in neural net uses the tanh activation function
        to bound answer within -1-1 range, introduces non-linearity
        '''
        # 3x2
        output = self.weights * parent_node.signal.reshape(-1, 1)

        # 3x2
        self.weights = self.activate(output)

        delta_input, delta_state = self.weights.T

        # TODO
        # HOW TO HANDLE VALUES OF VECTOR???
        delta_input = sum(np.dot(self.input, delta_input))/3
        delta_state = sum(np.dot(self.state, delta_state))/3
        self.signal = np.array([delta_input, delta_state, 1])

        self.input = delta_input
        self.state = delta_state
    # Activation Functions

    def activate(self, x: np.ndarray) -> np.ndarray:
        """The activation function (tanh)."""
        return np.tanh(x)

    def activate_derivative(self, x: np.ndarray) -> np.ndarray:
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
    n2 = Neuron(1.08)
    n1.feed_forward(n2)
    n2.feed_forward(n1)
