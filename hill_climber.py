#!/usr/local/bin python3

"""The Hill Climber.

As defined here: https://www.reddit.com/r/ludobots/wiki/core01
"""

import random
import numpy as np

__author__ = "Anass Al-Wohoush"


def create_matrix(rows, columns):
    """Generates random rows x columns matrix.
    All elements are numbers between 0 and 1.

    Args:
        rows: Number of rows.
        columns: Number of columns.

    Returns:
        Numpy matrix.
    """
    return np.matrix([
        [random.random() for y in range(columns)]
        for x in range(rows)
    ])


def perturb(v, prob):
    """Returns a perturbed copy of a matrix.
    
    Args:
        v: Numpy matrix to perturb.
        prob: Probability of perturbing each element.

    Returns:
        Perturbed numpy matrix.
    """
    child = v.copy()
    rows, columns = v.shape
    for i in range(rows):
        for j in range(columns):
            if prob > random.random():
                child[i, j] = random.random()
    return child


def climb(v, fitness, generations=5000, perturbations=0.05):
    """Yields most fit matrix of a generation.

    Args:
        v: Starting numpy matrix.
        fitness: Fitness function of type numpy matrix to float.
        generations: Number of generations, default: 5000.
        perturbations: Probability of perturbing an element, default: 5%.

    Yields:
        Tuple of current most fit matrix and its fitness.
    """
    parent = v
    parent_fitness = fitness(parent)
    yield parent, parent_fitness

    for gen in range(generations):
        child = perturb(parent, perturbations)
        child_fitness = fitness(child)
        if child_fitness > parent_fitness:
            parent = child
            parent_fitness = child_fitness
        yield parent, parent_fitness


if __name__ == "__main__":
    from matplotlib import cm
    import matplotlib.pyplot as plt

    # Part a.
    v = create_matrix(1, 50)
    hill = [fitness for p, fitness in climb(v, np.mean)]
    plt.plot(hill)
    plt.show()

    # Part b.
    for i in range(5):
        v = create_matrix(1, 50)
        hill = [fitness for p, fitness in climb(v, np.mean)]
        plt.plot(hill)
    plt.show()

    # Part c.
    v = create_matrix(1, 50)
    genes = np.matrix([p.tolist()[0] for p, fitness in climb(v, np.mean)]).T
    plt.imshow(genes, cmap=cm.gray, aspect="auto", interpolation="nearest")
    plt.show()
