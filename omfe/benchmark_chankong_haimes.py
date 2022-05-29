import matplotlib.pyplot as plt
import numpy as np

# Important for Optimization with pymoo:
# * Objective functions are always minimized => Multiply maximize objectives with -1
# * Constraints must be of type <= 0 => Reformulate
# * Normalize constraints to same scale => divide constraint by value at 0

# Three ways of defining problems:
# ElementWise, Vectorized or Functional

# List of Test-Functions on Wikipedia (https://en.wikipedia.org/wiki/Test_functions_for_optimization)
# A `+` indicating that this is already available in pymoo
# + Binh and Korn (BNH)
# * Chankong and Haimes
# * Fonseca-Fleming
# * Test function 4
# + Kursawe
# + Schaffer function #1 (go-schaffer01)
# + Schaffer function #2 (go-schaffer02)
# * Poloni's two objective function
# + Zitzler-Deb-Thiele's (ZDT) function #1
# + Zitzler-Deb-Thiele's (ZDT) function #2
# + Zitzler-Deb-Thiele's (ZDT) function #3
# + Zitzler-Deb-Thiele's (ZDT) function #4
# + Zitzler-Deb-Thiele's (ZDT) function #6
# + Osyczka and Kundu (OSY)
# + CTP1
# * Constr-Ex problem
# * Viennet function

from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.visualization.scatter import Scatter

# Predefined Problems, Algorithms and Terminations
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.termination.default import MultiObjectiveDefaultTermination


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


class BihnKorn(ElementwiseProblem):
    """Definition of the Chankong and Haimes function/problem

    See https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """

    def __init__(self):
        super().__init__(
            n_var=2, n_obj=2, n_constr=2, xl=np.array([0, 0]), xu=np.array([5, 3])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 4 * (x[0] ** 2) + 4 * (x[1] ** 2)
        f2 = (x[0] - 5) ** 2 + (x[1] - 5) ** 2

        # Rearranged to "<= 0 constraint" and normalized
        g1 = 1 * ((x[0] - 5) ** 2 + x[1] ** 2 - 25)
        g2 = 1 * (7.7 - (x[0] - 8) ** 2 - (x[1] + 3) ** 2)

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


def main():
    """Uses the NSGA2 algorithm to aproximate the pareto front on the Chankong
    and Haimes Function

    NSGA2 uses
      * Binary Tournament Selection (Drawing randomly two individuals, choosing better rank)
      * Simulated Binary Crossover (SBX)
      * Polynomial Mutation (PM)
      * Then merge parents and offspring population and select "best ranking ones"
      * Ranking:
        1. Non-dominated sorting (Solely Pareto-Order)
        2. Same-rank individuals are ranked by crowding distance (as measure of diversity)
    """
    problem = ChankongHaimes()
    algorithm = NSGA2()
    termination = MultiObjectiveDefaultTermination()
    result = minimize(
        problem,
        algorithm,
        termination,
        pf=True,
        seed=1,
        save_history=True,
        verbose=True,
    )
    plot = Scatter(title="Chankong and Haimes Function")
    plot.add(result.F, facecolor="none", edgecolor="red", alpha=0.8, s=20)
    plot.show()


if __name__ == "__main__":
    main()
