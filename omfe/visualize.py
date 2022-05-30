import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.typing import NDArray

from .problems import Problem

from .problems import bnh, ch
from .mga import MicroGeneticAlgorithm


def plot_agents(
    axes: mpl.axes.Axes, agents: NDArray[np.float64], problem: Problem, param_dict
):
    """Takes a list of agents for a given problem and plots them, returning the plot"""

    # TODO: Highlight current front! use ranking for this
    # TODO: cleanup mga, mga alone and then wrapper that calls it multiple times ('random_restarts')

    f1 = []
    f2 = []
    for agent in agents:
        function_values = problem.evaluate_functions(agent)
        f1.append(function_values[0])
        f2.append(function_values[1])

    axes.set_xlabel("f1")
    axes.set_ylabel("f2")
    axes.set_title(f"Objective Space Plot: {problem}")
    out = axes.scatter(f1, f2, **param_dict)
    return out
