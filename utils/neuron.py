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
        self.weights = np.random.rand(2, 3)
        # self.weights = np.array([[np.random.normal(mu, sigma)
        #                           for _ in range(2)] for _ in range(3)])
        # -1 adds direction to magnituide of vector state
        self.signal = np.array(
            [input, np.random.normal(mu, sigma), 1]).reshape(-1, 1)
        # Introduce Time :)
        self.state = np.random.randint(-1, 1)

    # Training
    def feed_forward(self, parent_node):
        '''
        Feedforward in neural net uses the tanh activation function
        to bound answer within -1-1 range, introduces non-linearity
        '''
        # 2x3 * 3x1
        if len(self.weights) != 2:
            self.weights = list(np.array(self.weights).T)
        output = list(self.activate(np.dot(self.weights, parent_node.signal)))
        self.weights = [[self.input, self.state], [
            parent_node.input, parent_node.state], [output[0][0], output[1][0]]]

        # 2x3
        delta_i, delta_s = output
        self.input = delta_i[0]
        self.state = delta_s[0]
        # 3x1
        self.signal = np.array([self.input, self.state, 1]).reshape(-1, 1)
        # recurrent (system dynamics by feeding input -> output -> input)
        # 2x3
        print(
            f'output {output} and weights {self.weights} and state {self.state}')
        print('')

    def backprop(self):
        pass

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
        return self.weights

    def set_weights(self, vector: list):
        '''
        set weights to vector
        '''
        self.weights = vector


if __name__ == "__main__":
    n1 = Neuron(1.089)
    n2 = Neuron(n1.input)
    n1.feed_forward(n2)
    n2.feed_forward(n1)
    print(n1.state)
    print(n2.state)
