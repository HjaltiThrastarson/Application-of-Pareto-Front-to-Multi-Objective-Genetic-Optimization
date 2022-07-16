"""Module containing various MOO test problems"""
from typing import Callable, Sequence
import numpy as np
import numpy.typing as npt

# TODO: If too slow, vectorize evaluate_functions and is_inside_constraints
# i.e. implement it for single agents and lists of agents directly for a problem
# instead of a general iterator


class Problem:
    """Default problem class consisting of functions, constraints and the search domain"""

    def __init__(
        self,
        functions: Sequence[Callable],
        constraints: Sequence[Callable],
        search_domain: Sequence[Sequence[int]],
    ) -> None:
        self.functions = functions
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

    @property
    def num_variables(self) -> int:
        return len(self.search_domain)

    @property
    def num_objectives(self) -> int:
        return len(self.functions)


class ChankongHaimes(Problem):
    def __init__(self) -> None:
        super().__init__(
            functions=[self.f_1, self.f_2],
            constraints=[self.g_1, self.g_2],
            search_domain=[[-20, 20], [-20, 20]],
        )

    def __repr__(self) -> str:
        return "ChankongHaimes"

    def __str__(self) -> str:
        return "Chankong and Haimes function"

    @staticmethod
    def f_1(agent: Sequence[float]) -> float:
        return 2 + (agent[0] - 2) ** 2 + (agent[1] - 1) ** 2

    @staticmethod
    def f_2(agent: Sequence[float]) -> float:
        return 9 * agent[0] - (agent[1] - 1) ** 2

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
            constraints=[self.g_1, self.g_2],
            search_domain=[[0, 5], [0, 3]],
        )

    def __repr__(self) -> str:
        return "BinhKorn"

    def __str__(self) -> str:
        return "Binh and Korn function"

    @staticmethod
    def f_1(agent: Sequence[float]) -> float:
        return 4 * (agent[0] ** 2) + 4 * (agent[1] ** 2)

    @staticmethod
    def f_2(agent: Sequence[float]) -> float:
        return (agent[0] - 5) ** 2 + (agent[1] - 5) ** 2

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
            constraints=[],
            search_domain=[[-5, 5], [-5, 5], [-5, 5]],
        )

    def __repr__(self) -> str:
        return "Kursawe"

    def __str__(self) -> str:
        return "Kursawe function"

    @staticmethod
    def f_1(agent: Sequence[float]) -> float:
        return (-10 * np.exp(-0.2 * np.sqrt(agent[0] ** 2 + agent[1] ** 2))) + (
            -10 * np.exp(-0.2 * np.sqrt(agent[1] ** 2 + agent[2] ** 2))
        )

    @staticmethod
    def f_2(agent: Sequence[float]) -> float:
        return (
            (np.abs(agent[0]) ** 0.8 + 5 * np.sin(agent[0] ** 3))
            + (np.abs(agent[1]) ** 0.8 + 5 * np.sin(agent[1] ** 3))
            + (np.abs(agent[2]) ** 0.8 + 5 * np.sin(agent[2] ** 3))
        )


class CTP1(Problem):
    def __init__(self) -> None:
        super().__init__(
            functions=[self.f_1, self.f_2],
            constraints=[self.g_1, self.g_2],
            search_domain=[[0, 1], [0, 1]],
        )

    def __repr__(self) -> str:
        return "CTP1"

    def __str__(self) -> str:
        return "CTP1"

    @staticmethod
    def f_1(agent: Sequence[float]) -> float:
        return agent[0]

    @staticmethod
    def f_2(agent: Sequence[float]) -> float:
        return (1 + agent[1]) * np.exp(-agent[0] / (1 + agent[1]))

    @staticmethod
    def g_1(agent: Sequence[float]) -> float:
        return CTP1.f_2(agent) / (0.858 * np.exp(-0.541 * CTP1.f_1(agent))) >= 1

    @staticmethod
    def g_2(agent: Sequence[float]) -> float:
        return CTP1.f_2(agent) / (0.728 * np.exp(-0.295 * CTP1.f_1(agent))) >= 1
