import numpy as np
from numpy import array, count_nonzero, random, ones, linspace
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self):
        random.seed(1)
        self.w = random.random((2, 1)) * 2 - 1
        self.wb = random.random(1) * 2 - 1

    def train(self, training_data, teacher, iterations):
        e = ones(training_data.shape[0]) - 2

        for i in range(iterations):
            for j in range(training_data.shape[0]):
                y = self.classify(training_data[j])
                e[j] = teacher[j] - y
                if count_nonzero(e) == 0:
                    i = iterations
                    break
                if e[j] == 0:
                    continue

                w0_delta = e[j] * training_data[j][0]
                w1_delta = e[j] * training_data[j][1]
                wb_adjustment = e[j]
                self.w[0] += w0_delta
                self.w[1] += w1_delta
                self.wb += wb_adjustment

    def classify(self, input_data):
        return 1 if (input_data[0] * self.w[0] + input_data[1] * self.w[1] + self.wb) > 0 else 0


def draw_line(p1, p2):
    a = (p1[1] - p2[1]) / (p1[0] - p2[0])
    b = p1[1] - a * p1[0]
    x_axis = linspace(-300, 300, 6000)
    y_axis = a * x_axis + b
    plt.xlim(-20, 20)
    plt.ylim(-5, 30)
    plt.plot(x_axis, y_axis)


if __name__ == '__main__':
    training_input = array(
        [[0, 23], [0, 4], [13, 7], [11, 19], [7, 12], [6, -1], [13, 11], [8, 11], [4, 8], [-9, 31],
         [-15, 13], [-5, 12], [-6, 24], [7, 16], [-2, 17], [-9, 17], [-7, 15], [15, 8], [6, 3], [-4, 14], [-5, 21],
         [6, 6], [-3, 18], [12, 3], [-8, 21], [-4, 13], [-5, 22], [5, 2], [7, 5], [5, 9], [0, 10],
         [3, 17], [-2, 9], [-5, -3]])
    classes = array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1,
                     1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0,
                     1, 1, 1, 0])
    training_output = classes.reshape((1, classes.shape[0])).T

    neuron = Neuron()
    print(f'Weights before learning: w1 - {neuron.w[0]}, w2 - {neuron.w[1]}, wb - {neuron.wb}')
    neuron.train(training_input, training_output, 100)
    print(f'Weights after learning: w1 - {neuron.w[0]}, w2 - {neuron.w[1]}, wb - {neuron.wb}')

    p1 = [0, -neuron.wb/neuron.w[1]]
    p2 = [-neuron.wb/neuron.w[0], 0]
    draw_line(p1, p2)
    plt.plot(training_input[:, 0][classes == 0], training_input[:, 1][classes == 0], 'b^')
    plt.plot(training_input[:, 0][classes == 1], training_input[:, 1][classes == 1], 'ro')
    # Neuron generated points
    plt.scatter(p1[0], p1[1], marker='D', c=[[0x00ff00]])
    plt.scatter(p2[0], p2[1], marker='D', c=[[0x00ff00]])
    plt.show()
