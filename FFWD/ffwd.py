import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    print('ANN is learning...')

    # Creating training set
    x = np.linspace(0, 9, 30)
    y1 = 2 * x * np.sin(x)
    coef = np.abs(max(y1) - min(y1))
    y = y1 / coef

    size = len(x)
    inputs = x.reshape(size, 1)
    target = y.reshape(size, 1)

    # Preparing some additional variables for plotting
    hidden_layer_neurons = [3, 5, 10, 15, 30, 50]
    training_methods_names = ['train_gd', 'train_gdm', 'train_gda', 'train_gdx', 'train_rprop']
    colors = ['r', 'g--', 'b', 'y--', 'c']

    # Creating ANNs with two layers, each having different amount of neurons,
    # initialized with random values, in hidden layer, and 1 neuron in output layer
    ANNs = [nl.net.newff([[-7, 7]], [n, 1]) for n in hidden_layer_neurons]
    training_methods = [nl.train.train_gd, nl.train.train_gdm, nl.train.train_gda, nl.train.train_gdx,
                        nl.train.train_rprop]

    rows = len(ANNs)
    errors = np.zeros((rows, 5))
    i = j = 0

    for net in ANNs:
        for tm in training_methods:
            # Training and testing ANN
            net.trainf = tm
            errors[i, j] = min(net.train(inputs, target, epochs=1000, show=0, goal=0.01))
            out = net.sim(inputs).reshape(size)

            # Plotting trained ANN and real values
            plt.plot(x, out, f'{colors[j]}')
            plt.title(f'Neurons: {hidden_layer_neurons[i]}')
            j += 1

        plt.plot(x, y, '.k')
        plt.legend([f'{training_methods_names[0]}', f'{training_methods_names[1]}', f'{training_methods_names[2]}',
                    f'{training_methods_names[3]}', f'{training_methods_names[4]}', 'Actual values'])
        plt.show()
        i += 1
        j = 0

    # Displaying error values for all teaching methods
    df = pd.DataFrame(errors, hidden_layer_neurons, training_methods_names)
    print()
    print(7 * ' ' + 'Error values for different teaching methods')
    print(58 * '-')
    print(df)
