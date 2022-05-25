"""Module containing various MOO test problems"""
from typing import List, Sequence


class Problem:
    """Default problem class consisting of functions, constraints and the search domain"""

    def __init__(self, functions, constraints, search_domain) -> None:
        self.functions = functions
        self.constraints = constraints
        self.search_domain = search_domain

    def evaluate_functions(self, agent) -> List[float]:
        return [fun(agent) for fun in self.functions]

    def evaluate_constraints(self, agent) -> List[bool]:
        return [constr(agent) for constr in self.constraints]

    @property
    def num_variables(self):
        return len(self.search_domain)

    @property
    def num_objectives(self):
        return len(self.functions)


class ChankongHaimes(Problem):
    def __init__(self) -> None:
        super().__init__(
            functions=[self.f_1, self.f_2],
            constraints=[self.g_1, self.g_2],
            search_domain=[[-20, 20], [-20, 20]],
        )

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


ch = ChankongHaimes()
bnh = BinhKorn()
