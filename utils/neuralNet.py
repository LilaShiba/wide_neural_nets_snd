import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from neuron import Neuron
from layer import Layer


class NetRunner:
    def __init__(self, name: str, layer_sizes: List[int], net_type: int) -> None:
        """
        Initialize the neural network.

        Parameters:
        - name: Name of the neural network.
        - layer_sizes: A list containing the sizes of each layer in the network.
        - net_type: An integer representing the type of network.
        """
        self.name: str = name
        self.layer_sizes: List[int] = layer_sizes
        self.type: str = self._get_net_type(net_type)

        self.neurons: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.create_network()

    def _get_net_type(self, net_type: int) -> str:
        """
        Retrieve the type of network based on the provided integer.

        Parameters:
        - net_type: An integer representing the type of network.

        Returns:
        - A string representing the type of network.
        """
        types = {
            1: 'cot',
            2: 'matrix',
            3: 'corpus',
            4: 'snd'
        }
        return types.get(net_type, 'unknown')

    def create_network(self) -> None:
        """
        Initialize the weights and biases for each layer in the network.
        """
        # Loop through each layer and initialize weights and biases
        for i, v in enumerate(self.layer_sizes):
            # init new neuron as matrix
            self.neurons.append(Layer(i, v))

        print(self.neurons)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Parameters:
        - z: Weighted sum matrix.

        Returns:
        - Activation matrix.
        """
        # The sigmoid function maps any value to a value between 0 and 1
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Derivative of the sigmoid activation function.

        Parameters:
        - z: Weighted sum matrix.

        Returns:
        - Derivative matrix.
        """
        # Derivative of sigmoid function is used during backpropagation step
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feedforward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the network.

        Parameters:
        - X: Input matrix.

        Returns:
        - Output matrix after forward pass.
        """
        self.a = [X]  # Store the input matrix
        self.z = []   # Initialize an empty list to store weighted sums

        # Loop through each layer, calculate weighted sum and activation

        for n, b in zip(self.neurons, self.biases):
            i = 0
            d_l = []
            for j in range(len(self.biases[i])):

                # Weighted sum
                n_delta = Neuron('delta', j)
                n_delta.set_params = self.a[-1][-1] + b[-1][-1]
                print(n_delta.neuron)
                self.z.append(n.update(n_delta))
                print(n.state)
                d_l.append(n.state)   # Activation
                i += 1

        return self.a[-1]  # Return the final output

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float, backpropagate: bool = True) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Train the neural network for a specified number of epochs.

        Parameters:
        - X: Input matrix.
        - y: Target output matrix.
        - epochs: Number of training epochs.
        - learning_rate: Learning rate for weight and bias updates.

        Returns:
        - Dictionary containing loss and parameters at each epoch.
        """
        output_dict = {}

        for epoch in range(epochs):
            y_pred = self.feedforward(X)
            # loss = self.compute_loss(y_pred, y)

            # if backpropagate:
            #     self.backpropagate(X, y, learning_rate)

            # output_dict[epoch] = {
            #     "loss": loss,
            #     "weights": [w.copy() for w in self.weights],
            #     "biases": [b.copy() for b in self.biases]
            # }

        return y_pred

    def plot_training_output(self, output_dict: Dict[int, Dict[str, np.ndarray]]) -> None:
        """
        Plot the training loss over epochs using data from output_dict.

        Parameters:
        - output_dict: Dictionary containing loss and parameters at each epoch.
        """
        epochs = list(output_dict.keys())
        loss_values = [output_dict[epoch]['loss'] for epoch in epochs]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b')
        plt.title(f"Training Loss over Epochs ({self.name})")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    mu = 0          # mean
    sigma = 1       # standard deviation
    input_size = 10  # col: features
    train_size = 1000  # row: number of training examples
    out_put_size = 3  # anwsers

    # Using normal distribution
    X_train = np.random.normal(mu, sigma, (input_size, train_size))
    # 1 output, 1000 examples
    y_train = np.random.normal(mu, sigma, (out_put_size, train_size))

    # Initialize a wide network: [10, 100, 1] means an input layer of size 10,
    # a hidden layer of size 100, and an output layer of size 1
    wide_network = NetRunner(name="WideNetworkMatrix", layer_sizes=[
        input_size, 128, 64, out_put_size], net_type=2)

    # Train the wide network
    # output_dict_wide = wide_network.train(
    #    X_train, y_train, epochs=12, learning_rate=0.01)

    # Plot the training loss for the wide network
    # wide_network.plot_training_output(output_dict_wide)

    # # Deep wide

    # deep_wide_network = NetRunner(name="DeepWideNetwork", layer_sizes=[
    #     input_size, 10000, 10000, 10000, 10000, 10000, 10000, 10000, out_put_size], net_type=2)

    # # Train the wide network
    # output_dict_wide = deep_wide_network.train(
    #     X_train, y_train, epochs=12, learning_rate=0.01)

    # # Plot the training loss for the wide network
    # deep_wide_network.plot_training_output(output_dict_wide)

    # Initialize a deep network: [10, 5, 5, 1] means an input layer of size 10,
    # two hidden layers of size 5, and an output layer of size 1
    # deep_network = NetRunner(name="DeepNetwork", layer_sizes=[
    #     input_size, 5, 5, 5, 5, 5, 5, 5, 5, out_put_size], net_type=3)

    # # Train the deep network
    # output_dict_deep = deep_network.train(
    #     X_train, y_train, epochs=12, learning_rate=0.01)

    # # Plot the training loss for the deep network
    # deep_network.plot_training_output(output_dict_deep)

    # scalar, vector, scalar, vector
    # state, neuron, state, neuron
