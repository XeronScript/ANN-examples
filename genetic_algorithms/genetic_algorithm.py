from typing import Callable
from math import sin, pi, sqrt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def adaptation(x: int) -> float:
    return 0.2 * sqrt(x) + 2 * sin(2 * pi * 0.02 * x) + 5


def roulette_wheel_selection(population: list[int], population_scores: list[float]):
    population_fitness = sum(population_scores)
    chromosome_probabilities = [score / population_fitness for score in population_scores]
    return np.random.choice(population, p=chromosome_probabilities)


def crossover(ch1: int, ch2: int, crossover_p: float) -> tuple[int, int]:
    # Crossover of two chromosomes recipe
    # ch1 ^ ((ch1^ch2) & locus)
    if np.random.random() < crossover_p:
        fewer_bits = min(np.ceil(np.log2(ch1 + 1)).astype(int), np.ceil(np.log2(ch2 + 1)).astype(int))
        crossing_point = int((np.random.random() * fewer_bits) + 1)
        mask = int(2 ** crossing_point - 1)
        ch1, ch2 = (ch1 ^ ((ch1 ^ ch2) & mask)), (ch2 ^ ((ch1 ^ ch2) & mask))

    return ch1, ch2


def mutation(ch: int, mutation_p: float) -> int:
    bits = np.ceil(np.log2(ch + 1))
    if np.random.random() < mutation_p and bits != 0:
        power = np.random.randint(bits)
        mutation_index = 2 ** power
        ch = ch ^ mutation_index
    return ch


def genetic_algorithm(evaluation_fun: Callable[[int], float], generations_num: int,
                      chromosomes_num: int, x_max: int, crossover_p: int, mutation_p: int) -> tuple[float, float]:
    # Generating first population
    population = [np.random.randint(x_max, dtype=np.uint8) for _ in range(chromosomes_num)]
    best, best_eval = 0, evaluation_fun(population[0])

    # Single population average adaptation
    avg_populations_adaptation = np.zeros(generations_num)
    j = 0

    # Iterating over all generations
    for gen in range(generations_num):
        # Calculating adaptation coefficient for all chromosomes
        scores = [evaluation_fun(x) for x in population]
        avg_populations_adaptation[j] = np.average(scores)
        j += 1

        for i in range(chromosomes_num):
            if scores[i] > best_eval:
                best, best_eval = population[i], scores[i]

        # Selecting chromosomes based on their adaptation coefficients
        selected_chromosomes = [roulette_wheel_selection(population, scores) for _ in range(chromosomes_num)]

        # Creating new generation
        children = []
        for i in range(0, chromosomes_num, 2):
            c1, c2 = selected_chromosomes[i], selected_chromosomes[i + 1]
            for c in crossover(c1, c2, crossover_p):
                c = mutation(c, mutation_p)
                children.append(c)
        population = children

    # Whole population average adaptation
    whole_pop_avg_adapt = np.average(avg_populations_adaptation)

    return whole_pop_avg_adapt, avg_populations_adaptation


if __name__ == '__main__':
    # [pm, pk] -> [mutation probability, crossover probability]
    pkm = [[[0,    0.5], [0,    0.6], [0,    0.7], [0,    0.8], [0,    1]],
           [[0.01, 0.5], [0.01, 0.6], [0.01, 0.7], [0.01, 0.8], [0.01, 1]],
           [[0.06, 0.5], [0.06, 0.6], [0.06, 0.7], [0.06, 0.8], [0.06, 1]],
           [[0.1,  0.5], [0.1,  0.6], [0.1,  0.7], [0.1,  0.8], [0.1,  1]],
           [[0.2,  0.5], [0.2,  0.6], [0.2,  0.7], [0.2,  0.8], [0.2,  1]],
           [[0.3,  0.5], [0.3,  0.6], [0.3,  0.7], [0.3,  0.8], [0.3,  1]],
           [[0.5,  0.5], [0.5,  0.6], [0.5,  0.7], [0.5,  0.8], [0.5,  1]]]

    generations = 100
    chromosomes = 50
    max_value = 255

    # Average adaptation coefficient of generations
    avg_coefficient_adaptation = np.ones((7, 5))
    # Rows, columns
    r = c = 0

    # Adaptation coefficient of all generations
    all_adapt_coeff = np.zeros((7, 5, generations))

    # All genetic algorithm iterations
    for row in pkm:
        for column in row:
            pm, pk = column
            avg_coefficient_adaptation[r, c], all_adapt_coeff[r, c] = \
                genetic_algorithm(adaptation, generations, chromosomes, max_value, pk, pm)
            c += 1
        r += 1
        c = 0

    # Creating a dataframe from gathered information
    df = pd.DataFrame(avg_coefficient_adaptation, columns=[pk[1] for pk in pkm[0]], index=[pm[0][0] for pm in pkm])
    print('pm\\pk')
    print(df)

    # Plotting gathered information
    x = np.arange(generations+1)
    i = j = 0
    for row in all_adapt_coeff:
        for y in row:
            plt.plot(x[:-1], y, label=f'pk = {pkm[0][j][1]}')
            plt.legend(loc='lower right')
            plt.title(f'pm = {pkm[i][0][0]}')
            plt.xlabel('Generations')
            plt.ylabel('Average Adaptation Coefficient')
            j += 1
        plt.show()
        i += 1
        j = 0
