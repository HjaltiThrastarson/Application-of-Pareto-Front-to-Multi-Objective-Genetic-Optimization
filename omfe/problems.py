"""Module containing various MOO test problems"""
from abc import ABC, abstractmethod
from typing import Callable, Sequence
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from omfe.benchmark import (
    plot_binh_korn_pareto_front,
    plot_chankong_haimes_pareto_front,
    plot_ctp1_pareto_front,
    plot_kursawe_pareto_front,
)

# TODO: If too slow, vectorize evaluate_functions and is_inside_constraints
# i.e. implement it for single agents and lists of agents directly for a problem
# instead of a general iterator


class Problem(ABC):
    """Default problem class consisting of functions, constraints and the search domain"""

    def __init__(
        self,
        functions: Sequence[Callable],
        functions_vec: Sequence[Callable],
        constraints: Sequence[Callable],
        search_domain: Sequence[Sequence[int]],
    ) -> None:
        self.functions = functions
        self.functions_vec = functions_vec
        self.constraints = constraints
        self.search_domain = search_domain

    def evaluate_functions(
        self, agent: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluate all objective functions for the given agent"""
        return np.array([fun(agent) for fun in self.functions])

    def is_inside_constraints(self, agent: npt.NDArray[np.float64]) -> bool:
        """Returns True if agent is within problem constraints, False otherwise"""
        constraints_met: bool = np.all(
            np.array([constr(agent) for constr in self.constraints])
        ).astype(bool)
        return constraints_met

    def breaks_constraint(self, agent: npt.NDArray[np.float64]) -> bool:
        """Returns True if any number of constraints is broken"""
        return not self.is_inside_constraints(agent)

    @staticmethod
    @abstractmethod
    def plot_pareto_front(axes: plt.Axes, *args, **kwargs):
        pass

    @property
    def num_variables(self) -> int:
        return len(self.search_domain)

    @property
    def num_objectives(self) -> int:
        return len(self.functions)


def plot_agents(
    axes: plt.Axes,
    agents: npt.NDArray[np.float64],
    problem: Problem,
    *args,
    **kwargs,
):
    """Takes a list of agents for a given problem and plots them, returning the plot"""

    # TODO: Highlight current front! use pareto_front from evaluator for this
    # TODO: Move this to another more fitting module

    F = np.empty((len(agents), 2))
    for i, agent in enumerate(agents):
        F[i] = problem.evaluate_functions(agent)

    axes.set_xlabel("f1")
    axes.set_ylabel("f2")
    axes.set_title(f"Objective Space Plot: {problem}")
    return axes.scatter(F[:, 0], F[:, 1], *args, **kwargs)


class ChankongHaimes(Problem):
    def __init__(self) -> None:
        super().__init__(
            functions=[self.f_1, self.f_2],
            functions_vec=[self.f_1_vec, self.f_2_vec],
            constraints=[self.g_1, self.g_2],
            search_domain=[[-20, 20], [-20, 20]],
        )

    def __repr__(self) -> str:
        return "ChankongHaimes"

    def __str__(self) -> str:
        return "Chankong and Haimes function"

    @staticmethod
    def plot_pareto_front(axes: plt.Axes, *args, **kwargs):
        return plot_chankong_haimes_pareto_front(axes, *args, **kwargs)

    @staticmethod
    def f_1(agent: Sequence[float]) -> float:
        return 2 + (agent[0] - 2) ** 2 + (agent[1] - 1) ** 2

    @staticmethod
    def f_1_vec(agents: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 2 + (agents[:, 0] - 2) ** 2 + (agents[:, 1] - 1) ** 2

    @staticmethod
    def f_2(agent: Sequence[float]) -> float:
        return 9 * agent[0] - (agent[1] - 1) ** 2

    @staticmethod
    def f_2_vec(agents: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 9 * agents[:, 0] - (agents[:, 1] - 1) ** 2

    @staticmethod
    def g_1(agent: Sequence[float]) -> bool:
        return (agent[0] ** 2 + agent[1] ** 2) <= 225

    @staticmethod
    def g_2(agent: Sequence[float]) -> bool:
        return (agent[0] - 3 * agent[1] + 10) <= 0


class BinhKorn(Problem):
    def __init__(self) -> None:
        super().__init__(
            functions=[self.f_1, self.f_2],
            functions_vec=[self.f_1_vec, self.f_2_vec],
            constraints=[self.g_1, self.g_2],
            search_domain=[[0, 5], [0, 3]],
        )

    def __repr__(self) -> str:
        return "BinhKorn"

    def __str__(self) -> str:
        return "Binh and Korn function"

    @staticmethod
    def plot_pareto_front(axes: plt.Axes, *args, **kwargs):
        return plot_binh_korn_pareto_front(axes, *args, **kwargs)

    @staticmethod
    def f_1(agent: Sequence[float]) -> float:
        return 4 * (agent[0] ** 2) + 4 * (agent[1] ** 2)

    @staticmethod
    def f_1_vec(agents: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 4 * (agents[:, 0] ** 2) + 4 * (agents[:, 1] ** 2)

    @staticmethod
    def f_2(agent: Sequence[float]) -> float:
        return (agent[0] - 5) ** 2 + (agent[1] - 5) ** 2

    @staticmethod
    def f_2_vec(agents: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (agents[:, 0] - 5) ** 2 + (agents[:, 1] - 5) ** 2

    @staticmethod
    def g_1(agent: Sequence[float]) -> bool:
        return (agent[0] - 5) ** 2 + agent[1] ** 2 <= 25

    @staticmethod
    def g_2(agent: Sequence[float]) -> bool:
        return (agent[0] - 8) ** 2 + (agent[1] + 3) ** 2 >= 7.7


class Kursawe(Problem):
    def __init__(self) -> None:
        super().__init__(
            functions=[self.f_1, self.f_2],
            functions_vec=[self.f_1_vec, self.f_2_vec],
            constraints=[],
            search_domain=[[-5, 5], [-5, 5], [-5, 5]],
        )

    def __repr__(self) -> str:
        return "Kursawe"

    def __str__(self) -> str:
        return "Kursawe function"

    @staticmethod
    def plot_pareto_front(axes: plt.Axes, *args, **kwargs):
        return plot_kursawe_pareto_front(axes, *args, **kwargs)

    @staticmethod
    def f_1(agent: Sequence[float]) -> float:
        return (-10 * np.exp(-0.2 * np.sqrt(agent[0] ** 2 + agent[1] ** 2))) + (
            -10 * np.exp(-0.2 * np.sqrt(agent[1] ** 2 + agent[2] ** 2))
        )

    @staticmethod
    def f_1_vec(agents: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (-10 * np.exp(-0.2 * np.sqrt(agents[:, 0] ** 2 + agents[:, 1] ** 2))) + (
            -10 * np.exp(-0.2 * np.sqrt(agents[:, 1] ** 2 + agents[:, 2] ** 2))
        )

    @staticmethod
    def f_2(agent: Sequence[float]) -> float:
        return (
            (np.abs(agent[0]) ** 0.8 + 5 * np.sin(agent[0] ** 3))
            + (np.abs(agent[1]) ** 0.8 + 5 * np.sin(agent[1] ** 3))
            + (np.abs(agent[2]) ** 0.8 + 5 * np.sin(agent[2] ** 3))
        )

    @staticmethod
    def f_2_vec(agents: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (
            (np.abs(agents[:, 0]) ** 0.8 + 5 * np.sin(agents[:, 0] ** 3))
            + (np.abs(agents[:, 1]) ** 0.8 + 5 * np.sin(agents[:, 1] ** 3))
            + (np.abs(agents[:, 2]) ** 0.8 + 5 * np.sin(agents[:, 2] ** 3))
        )


class CTP1(Problem):
    def __init__(self) -> None:
        super().__init__(
            functions=[self.f_1, self.f_2],
            functions_vec=[self.f_1_vec, self.f_2_vec],
            constraints=[self.g_1, self.g_2],
            search_domain=[[0, 1], [0, 1]],
        )

    def __repr__(self) -> str:
        return "CTP1"

    def __str__(self) -> str:
        return "CTP1 function"

    @staticmethod
    def plot_pareto_front(axes: plt.Axes, *args, **kwargs):
        return plot_ctp1_pareto_front(axes, *args, **kwargs)

    @staticmethod
    def f_1(agent: Sequence[float]) -> float:
        return agent[0]

    @staticmethod
    def f_1_vec(agents: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return agents[:, 0]

    @staticmethod
    def f_2(agent: Sequence[float]) -> float:
        return (1 + agent[1]) * np.exp(-agent[0] / (1 + agent[1]))

    @staticmethod
    def f_2_vec(agents: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (1 + agents[:, 1]) * np.exp(-agents[:, 0] / (1 + agents[:, 1]))

    @staticmethod
    def g_1(agent: Sequence[float]) -> float:
        return CTP1.f_2(agent) / (0.858 * np.exp(-0.541 * CTP1.f_1(agent))) >= 1

    @staticmethod
    def g_2(agent: Sequence[float]) -> float:
        return CTP1.f_2(agent) / (0.728 * np.exp(-0.295 * CTP1.f_1(agent))) >= 1
