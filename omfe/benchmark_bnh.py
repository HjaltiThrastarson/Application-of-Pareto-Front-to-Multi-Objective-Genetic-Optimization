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
from pymoo.visualization.scatter import Scatter
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video

# Predefined Problems, Algorithms and Terminations
from pymoo.problems.multi.bnh import BNH
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.termination.default import MultiObjectiveDefaultTermination


def main():
    problem = BNH()
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

    # Make convergence video with saved history
    with Recorder(Video("bnh.mp4")) as rec:
        for entry in result.history:
            plot = Scatter(title=f"BNH Gen {entry.n_gen}")
            plot.add(entry.pop.get("F"))
            plot.add(
                entry.problem.pareto_front(use_cache=False, flatten=False),
                plot_type="line",
                color="black",
                alpha=0.7,
            )
            plot.do()
            rec.record()

    # Plot end result
    plot = Scatter(title="BNH")
    plot.add(
        problem.pareto_front(use_cache=False, flatten=False),
        plot_type="line",
        color="black",
        alpha=0.7,
    )
    plot.add(result.F, facecolor="none", edgecolor="red", alpha=0.8, s=20)
    plot.show()


if __name__ == "__main__":
    main()
