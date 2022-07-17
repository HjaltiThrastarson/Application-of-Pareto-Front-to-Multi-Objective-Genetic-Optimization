"""Module to compute and plot benchmark results calculated with pymoo

This module contains functions to optimize various problems using the pymoo
optimization framework and to plot the results. This is intended as benchmark
for our own optimization algorithm and allows visual comparison of the results.

All functions use the NSGA2 algorithm for optimization which results in a list
of elements on the pareto front, that are then plotted as a line.
    
NSGA2
=====

* Binary Tournament Selection (Drawing randomly two individuals, choosing better rank)
* Simulated Binary Crossover (SBX)
* Polynomial Mutation (PM)
* Then merge parents and offspring population and select "best ranking ones"
* Ranking:
    1. Non-dominated sorting (Solely Pareto-Order)
    2. Same-rank individuals are ranked by crowding distance (as measure of diversity)

Pymoo
=====

* Objective functions are always minimized => Multiply maximize objectives with -1
* Constraints must be of type <= 0 => Reformulate
* Normalize constraints to same scale => divide constraint by value at 0
* Three ways of defining problems:
    - ElementWise
    - Vectorized
    - Functional

List of Test-Functions on Wikipedia (https://en.wikipedia.org/wiki/Test_functions_for_optimization)
A `+` indicating that this is already available in pymoo
+ Binh and Korn (BNH)
- Chankong and Haimes
- Fonseca-Fleming
- Test function 4
+ Kursawe
+ Schaffer function #1 (go-schaffer01)
+ Schaffer function #2 (go-schaffer02)
- Poloni's two objective function
+ Zitzler-Deb-Thiele's (ZDT) function #1
+ Zitzler-Deb-Thiele's (ZDT) function #2
+ Zitzler-Deb-Thiele's (ZDT) function #3
+ Zitzler-Deb-Thiele's (ZDT) function #4
+ Zitzler-Deb-Thiele's (ZDT) function #6
+ Osyczka and Kundu (OSY)
+ CTP1
- Constr-Ex problem
- Viennet function
"""

import numpy as np
import matplotlib.pyplot as plt
import pymoo as moo
import pymoo.config as mooconf
import pymoo.core.problem as mooprob
import pymoo.factory as moofac
import pymoo.optimize as mooopt
from pymoo.algorithms.moo.nsga2 import NSGA2

mooconf.Config.show_compile_hit = False


class ChankongHaimes(mooprob.ElementwiseProblem):
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


def plot_chankong_haimes_pareto_front(axes: plt.Axes, *args, **kwargs):
    problem = ChankongHaimes()
    algorithm = NSGA2()
    termination = moo.factory.get_termination("default_multi")
    result = mooopt.minimize(
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


def plot_binh_korn_pareto_front(axes: plt.Axes, *args, **kwargs):
    problem = moofac.get_problem("bnh")
    front = problem.pareto_front()
    return axes.plot(front[:, 0], front[:, 1], color="red", alpha=0.7, *args, **kwargs)


def plot_kursawe_pareto_front(axes: plt.Axes, *args, **kwargs):
    problem = moofac.get_problem("kursawe")
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


def plot_ctp1_pareto_front(axes: plt.Axes, *args, **kwargs):
    problem = moofac.get_problem("ctp1")
    front = problem.pareto_front()
    front = front[front[:, 0].argsort()]
    return axes.plot(front[:, 0], front[:, 1], color="red", alpha=0.7, *args, **kwargs)
