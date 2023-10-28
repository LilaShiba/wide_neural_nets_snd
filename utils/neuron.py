import numpy as np
import typing
import matplotlib.pyplot as plt
import networkx as nx


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
        self.state = np.random.randint(-1, 1)
        # 3x2
        self.weights = np.random.rand(3, 2)
        # 1x3
        self.signal = np.array(
            [input, np.random.normal(mu, sigma), 1])
        # -1 adds direction to magnituide of vector state
        delta_input, delta_state = self.init_feed_forward(self.signal)

        print(self.signal)
        self.output = self.input = delta_input

    # Training

    def init_feed_forward(self, signal):
        '''
        Feedforward in neural net uses the tanh activation function
        to bound answer within -1-1 range, introduces non-linearity
        '''
        # 3x1
        signal = self.activate(np.cross(self.weights, signal), False)
        delta_input = signal[0][0]
        delta_state = signal[1][0]
        # 3x1
        self.signal = np.array([delta_input, delta_state, 1])
        # recurrent (system dynamics by feeding input -> output -> input)
        # print(
        #     f'state: {self.state} | signal {self.signal}')
        # print('')
        return delta_input, delta_state

    def feed_forward(self, signal):
        '''
        Feedforward in neural net uses the tanh activation function
        to bound answer within -1-1 range, introduces non-linearity
        '''

        output = list(self.activate(self.weights.T * signal, False))

        # TODO: HOW TO FINETUNE/UPDATE weights
        # self.weights = [[signal[0], signal[1]], [
        #     self.input, self.state], [output[0][0], output[1][0]]]

        self.weights = np.dot(self.weights, output)
        print(f'weights:{self.weights}')

        # Notice time here for delta
        # 2x3
        self.input = output[0][0]
        self.state = output[1][0]
        # 3x1
        self.signal = np.array([self.input, self.state, 1])
        # recurrent (system dynamics by feeding input -> output -> input)
        # 2x3
        print(
            f'state: {self.state} | signal {self.signal}')
        print('')
        return self

    def backprop(self):
        '''
        TODO: Create :)
        '''
        pass
    # Activation Functions

    def activate(self, x: np.ndarray, sig: bool = False) -> np.ndarray:
        """The activation function (tanh is default)."""
        if sig:
            return 1 + 1/np.exp(-x)
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
    n1.feed_forward(n2.signal[0])
    n2.feed_forward(n1.signal[0])
    n1.feed_forward(n2.signal[0])
    print(n1.state)
    print(n2.state)
    plt.plot(np.tanh(n1.signal), label='tahn')
    plt.plot(1 / (1 + np.exp(-n1.signal)), label='sigmoid')
    plt.legend()
    plt.show()
