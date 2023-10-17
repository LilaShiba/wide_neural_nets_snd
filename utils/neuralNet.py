import matplotlib.pyplot as plt
import numpy as np
from layer import Layer


class NetRunner:
    def __init__(self, layer_sizes: list):
        self.layers = []
        for size in layer_sizes:
            self.layers.append(Layer(size))

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def mse_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return ((predictions - targets) ** 2).mean()

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int) -> list:
        # Dummy training loop just to show losses. Actual training not implemented.
        losses = []
        for epoch in range(epochs):
            predictions = np.array([self.feedforward(xi) for xi in X])
            loss = self.mse_loss(predictions, y)
            losses.append(loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.5f}")
        return losses

    @staticmethod
    def plot_losses(losses: list) -> None:
        plt.plot(losses)
        plt.title("Training Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error (MSE) Loss")
        plt.show()


if __name__ == "__main__":
    # Example data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [
                 1, 1, 1, 1], [0, 0, 0, 0]])  # XOR problem

    # Deep network with three hidden layers
    deep_network = NetRunner([10, 5, 5, 2])

    # Wide network with one hidden layer having a lot of neurons
    wide_network = NetRunner([10, 100, 2])

    # Both wide and deep network
    deep_wide_network = NetRunner([10, 100, 100, 100, 2])

    # Train and plot
    losses = wide_network.train(X, y, epochs=10)
    NetRunner.plot_losses(losses)
