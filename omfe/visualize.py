"""Helper module for easier visualization and visual comparison of results"""

import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
from numpy.typing import NDArray

from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.problems.multi.bnh import BNH
from pymoo.problems.multi.kursawe import Kursawe
from pymoo.problems.multi.ctp import CTP1
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.config import Config
from omfe.problems import Problem


Config.show_compile_hint = False


class ChankongHaimes(ElementwiseProblem):
    """Definition of the Chankong and Haimes function/problem

    See https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """

    def __init__(self):
        super().__init__(
            n_var=2, n_obj=2, n_constr=2, xl=np.array([-20, -20]), xu=np.array([20, 20])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 2 + (x[0] - 2) ** 2 + (x[1] - 1) ** 2
        f2 = 9 * x[0] - (x[1] - 1) ** 2

        # Rearranged to "<= 0 constraint" and normalized
        g1 = 1 / 225 * (x[0] ** 2 + x[1] ** 2 - 225)
        g2 = 1 / 10 * (x[0] - 3 * x[1] + 10)

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


def plot_agents(
    axes: mpl.axes.Axes, agents: NDArray[np.float64], problem: Problem, *args, **kwargs
):
    """Takes a list of agents for a given problem and plots them, returning the plot"""

    # TODO: Highlight current front! use pareto_front from evaluator for this

    F = np.empty((len(agents), 2))
    for i, agent in enumerate(agents):
        F[i] = problem.evaluate_functions(agent)

    axes.set_xlabel("f1")
    axes.set_ylabel("f2")
    axes.set_title(f"Objective Space Plot: {problem}")
    return axes.scatter(F[:, 0], F[:, 1], *args, **kwargs)


def plot_chankong_haimes_pareto_front(axes: Axes, *args, **kwargs):
    problem = ChankongHaimes()
    algorithm = NSGA2()
    termination = MultiObjectiveDefaultTermination()
    result = minimize(
        problem,
        algorithm,
        termination,
    )
    F = result.F[result.F[:, 0].argsort()]
    return axes.plot(
        F[:, 0],
        F[:, 1],
        color="red",
        alpha=0.7,
        *args,
        **kwargs,
    )


def plot_binh_korn_pareto_front(axes: Axes, *args, **kwargs):
    problem = BNH()
    front = problem.pareto_front()
    return axes.plot(front[:, 0], front[:, 1], color="red", alpha=0.7, *args, **kwargs)


def plot_kursawe_pareto_front(axes: Axes, *args, **kwargs):
    problem = Kursawe()
    front = problem.pareto_front()
    front = front[front[:, 0].argsort()]
    return axes.plot(
        front[:, 0],
        front[:, 1],
        marker=".",
        color="red",
        alpha=0.7,
        linestyle="",
        *args,
        **kwargs,
    )


def plot_ctp1_pareto_front(axes: Axes, *args, **kwargs):
    problem = CTP1()
    front = problem.pareto_front()
    front = front[front[:, 0].argsort()]
    return axes.plot(front[:, 0], front[:, 1], color="red", alpha=0.7, *args, **kwargs)
