import numpy as np
from typing import List
import random


class Neuron:
    def __init__(self, input_val: float, state_val: float, output_val: float, newstate_val: float):
        """
        Initialize the neuron with its 2x2 matrix representation.

        Parameters:
        - input_val: Value for the input in the neuron's matrix.
        - state_val: Value for the state in the neuron's matrix.
        - output_val: Value for the output in the neuron's matrix.
        - newstate_val: Value for the new state in the neuron's matrix.
        """
        self.matrix = np.array([[input_val, state_val],
                                [output_val, newstate_val]])
        self.bias = 0.01

    def forward(self, input_matrix: np.ndarray) -> np.ndarray:
        """
        Feed-forward the input through the neuron.

        Parameters:
        - input_matrix: The input data as a numpy array.

        Returns:
        - The output after feeding the input through the neuron
        is a scalar
        """
        output = np.dot(input_matrix, self.matrix) + self.bias
        return self.activation(output)

    @staticmethod
    def activation(x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(x)

    def __str__(self) -> str:
        """String representation of the neuron's matrix."""
        return str(self.matrix)


class Layer:

    def __init__(self, neurons: List[int]) -> List[Neuron]:
        """
        Initialize a layer with a list of neurons.

        Parameters:
        - neurons: List of Neuron objects.

        """
        self.layers = []
        self.create(neurons)

    def create(self, neurons: List[int], replace: bool = True) -> List[Neuron]:
        '''
        Creates random neurons in each layer
        replace Optional:default True Bool, replace self.layers 
        '''
        res = []
        for idx, val in enumerate(neurons):
            for _ in range(1, val):
                # -1 adds direction to magnitude
                i = np.random.normal(-1, 1)
                s = np.random.normal(-1, 1)
                o = np.random.normal(-1, 1)
                ds = np.random.normal(-1, 1)
            res.append(Neuron(i, s, o, ds))
        if replace:
            self.layers = res
        return res

    def forward(self, input_matrix: np.ndarray) -> np.ndarray:
        """
        Feed-forward the input through the layer.

        Parameters:
        - input_matrix: The input data as a numpy array.

        Returns:
        - The output after feeding the input through the layer.
        """
        outputs = [neuron.forward(input_matrix) for neuron in self.layers]
        return np.array(outputs).T  # Transpose to match expected shape


class NeuralNetwork:
    def __init__(self, layers: List[int]):
        """
        Initialize the neural network with a list of layers.

        Parameters:
        - layers: List of Layer objects.
        """
        self.network = Layer(layers)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Feed the input data through all layers to get the prediction.

        Parameters:
        - input_data: The input data as a numpy array.

        Returns:
        - The prediction as a numpy array.
        """
        for layer in self.network.layers:
            # print(layer)
            # From layer to layer transfer data
            input_data = layer.forward(input_data)
        return input_data


if __name__ == "__main__":
    # Example usage:
    # neuron_A = Neuron(0.5, 0.2, 0.8, 0.4)
    # neuron_B = Neuron(0.1, 0.3, 0.5, 0.7)

    # layer1 = Layer([neuron_A, neuron_B])
    # layer2 = Layer([Neuron(0.2, 0.4, 0.6, 0.8), Neuron(0.9, 0.1, 0.3, 0.5)])

    network_size = 12
    nn = NeuralNetwork([1000]*network_size)

    input_data = np.array([[0.2, 0.5],
                           [-0.1, 0.1]])

    prediction = nn.predict(input_data)

    print("Input Data:")
    print(input_data)
    print("\nPrediction:")
    print(prediction)
