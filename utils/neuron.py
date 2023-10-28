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
    def feed_forward(self, signal):
        '''
        Feedforward in neural net uses the tanh activation function
        to bound answer within -1-1 range, introduces non-linearity
        '''
        # 2x3 * 3x1
        # ensure shapes for that ole dot product
        if len(self.weights) != 2:
            self.weights = list(np.array(self.weights).T)
        # list of scalars [input, state]
        output = list(self.activate(np.dot(self.weights, signal), False))

        # TODO: HOW TO FINETUNE/UPDATE weights
        # self.weights = [[signal[0], signal[1]], [
        #     self.input, self.state], [output[0][0], output[1][0]]]

        self.weights = self.weights * output
        print(f'weights:{self.weights}')

        # Notice time here for delta
        # 2x3
        self.input = output[0][0]
        self.state = output[1][0]
        # 3x1
        self.signal = np.array([self.input, self.state, 1]).reshape(-1, 1)
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
    n1.feed_forward(n2.signal)
    n2.feed_forward(n1.signal)
    n1.feed_forward(n2.signal)
    print(n1.state)
    print(n2.state)
    plt.plot(np.tanh(n1.signal), label='tahn')
    plt.plot(1 / (1 + np.exp(-n1.signal)), label='sigmoid')
    plt.legend()
    plt.show()
