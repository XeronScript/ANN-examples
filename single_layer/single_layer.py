from typing import Iterable
import numpy as np


class SingleLayer:

    def __init__(self) -> None:
        self.w = np.random.uniform(-0.1, 0.1, (5, 3))
        self.beta = 5
        self.learning_rate = 0.1

    def train(self, training_data: np.ndarray | Iterable, teacher: np.ndarray | Iterable, epoch: int) -> None:
        # Converting input lists if necessary
        if not isinstance(training_data, np.ndarray):
            training_data = np.array(training_data)
        if not isinstance(teacher, np.ndarray):
            teacher = np.array(teacher)

        for _ in range(epoch):
            # Teaching every neuron independently
            for i in range(training_data.shape[1]):
                # Extracting data for selected neuron
                x = training_data[:, i]
                x = x.reshape(x.size, 1)
                t = teacher[:, i]
                t = t.reshape(t.size, 1)

                # Calculating output of a neuron
                y = self.predict(x)

                # Calculating error and adjusting weights
                err = np.subtract(t, y)
                delta_w = self.learning_rate * x @ err.T
                self.w += delta_w

    def predict(self, x) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        # Calculating neuron output
        u = self.w.T @ x
        return 1 / (1 + np.exp(-self.beta * u + 1))

    def get_weights(self):
        return self.w


if __name__ == "__main__":
    # Preparing data
    inputs = np.array(
        [[4.0, 2.0, -1.0],
         [0.01, -1.0, 3.5],
         [0.01, 2.0, 0.01],
         [-1.0, 2.5, -2.0],
         [1.5, 2.0, 1.5]]
    )

    outputs = np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    ) # mammal | bird | fisch

    epoch = 100

    # Training ANN
    neurons = SingleLayer()
    neurons.train(inputs, outputs, epoch)

    # Testing ANN
    human = np.array([2, 0, 0, -1, -1.5])
    bat = np.array([2, -1, 1, -1, 1])
    shark = np.array([-1, 3, 0, -1, 1])
    human_prediction = neurons.predict(human)
    bat_prediction = neurons.predict(bat)
    shark_prediction = neurons.predict(shark)
    expected_human_result = np.array([1, 0, 0])
    expected_bat_result = np.array([1, 0, 0])
    expected_shark_result = np.array([0, 0, 1])

    np.set_printoptions(suppress=True)
    print(f'ANN human prediction: {np.around(human_prediction, decimals=6)}')
    print(f'Expected human result: {expected_human_result}')
    print(f'ANN bat prediction: {np.around(bat_prediction, decimals=6)}')
    print(f'Expected bat result: {expected_bat_result}')
    print(f'ANN shark prediction: {np.around(shark_prediction, decimals=6)}')
    print(f'Expected shark result: {expected_shark_result}')
