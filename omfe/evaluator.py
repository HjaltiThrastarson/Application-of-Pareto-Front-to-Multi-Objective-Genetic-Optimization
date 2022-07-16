"""Module containing classes/functions to evalute and sort agents based on different criteria"""
from abc import ABC, abstractmethod
from typing import Iterable, Set, Tuple, List, Any, Optional

import numpy as np
import numpy.typing as npt
from omfe.problems import Problem

# TODO: Use numpy instead of python inbuilt sets/lists
# TODO: Test weighted sum sort


class Evaluator(ABC):
    """Abstract Evaluator to evaluate agents and order them according to a
    specific ranking/scoring mechanism
    """

    def __init__(self, problem: Problem) -> None:
        self.problem = problem

    @abstractmethod
    def sort(self, agents: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Sort the given agents by fitness/ranking"""

    @abstractmethod
    def evaluate_sort(
        self, agents: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Sort agents by fitness/ranking, additionally returning the scoring

        Args:
            agents (npt.NDArray[np.float64]): List of agents (n-tuples). n being
            equals the number of variables of the given problem

        Returns:
            npt.NDArray[np.float64]: Two lists, one with the sorted agents, one
            with the fitness/ranking score
        """

    @abstractmethod
    def reset(self) -> None:
        """Function to call to reset the internal state of the evaluator"""

    @abstractmethod
    def is_better_than(
        self, agent_a: npt.NDArray[np.float64], agent_b: npt.NDArray[np.float64]
    ):
        """Returns true if agent_a is 'better' i.e. fitter/ranked higher than agent_b"""


class WeightBasedEvaluator(Evaluator):
    """Sort agents with n variables by creating a weighted sum of the individual
    objective functions. Lowest fitness score first. If no weights are provided
    a uniform random weight distribution with sum one will be used."""

    def __init__(
        self,
        problem: Problem,
        seed: int,
        weights: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        super().__init__(problem)
        self.rng = np.random.default_rng(seed)
        if weights:
            if len(self.problem.functions) != len(weights):
                raise ValueError(
                    "Weights Vector has wrong length! "
                    "Expected: {len(self.problem.functions)} Actual: {len(weights)}"
                )
            self.weights = weights
        else:
            self.weights = self._get_random_vec_with_sum_one(
                len(self.problem.functions)
            )

    def reset(self) -> None:
        self.weights = self._get_random_vec_with_sum_one(len(self.problem.functions))

    def evaluate_sort(
        self, agents: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        fitnesses = self._calculate_fitnesses_with_constraints(agents)
        fitnesses_sorted_idx = fitnesses.argsort()
        return agents[fitnesses_sorted_idx], fitnesses[fitnesses_sorted_idx]

    def sort(self, agents: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        fitnesses = self._calculate_fitnesses_with_constraints(agents)
        agents_by_fitness: npt.NDArray[np.float64] = agents[fitnesses.argsort()]
        return agents_by_fitness

    def is_better_than(
        self, agent_a: npt.NDArray[np.float64], agent_b: npt.NDArray[np.float64]
    ):
        return self._calculate_fitness_with_constraints(
            agent_a
        ) < self._calculate_fitness_with_constraints(agent_b)

    def _calculate_fitnesses_with_constraints(
        self, agents: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.array(
            [self._calculate_fitness_with_constraints(agent) for agent in agents]
        )

    def _calculate_fitness_with_constraints(
        self, agent: npt.NDArray[np.float64]
    ) -> np.float64:
        """Calculate the fitness of an agent, setting it to infinity if
        it breaks the problem constraints"""
        if self.problem.is_inside_constraints(agent):
            return self._calculate_fitness(agent)
        else:
            return np.float64("inf")

    def _calculate_fitness(self, agent: npt.NDArray[np.float64]) -> np.float64:
        """Calculates the fitness of an agent. Number of weights must be equal
        to number of objectives/problem functions"""
        fitness_list = self.problem.evaluate_functions(agent)
        weighted_fitness: np.float64 = np.dot(fitness_list, self.weights)
        return weighted_fitness

    def _get_random_vec_with_sum_one(self, length: int) -> npt.NDArray[np.float64]:
        """Generate a vector that sums up to 1 with uniform distribution

        This can be imagined as cutting a string at random locations and measuring
        the distance of the resulting pieces, thus the ends 0 and 1 need to be added.

        Note: This is NOT choosing every coordinate randomly and then rescaling
        to [0,1], as this would not result in a uniform distribution. See
        https://stackoverflow.com/a/8068956 for an explanation attempt.
        """
        cuts = np.concatenate([0, self.rng.random(size=length - 1), 1], axis=None)
        cuts.sort()
        return np.diff(cuts)

    def __str__(self) -> str:
        return f"Weights: {self.weights}"


class NonDominatedSortEvaluator(Evaluator):
    """Sort agents with n variables according to non-dominance of the given
    problem functions
    """

    def reset(self) -> None:
        return None

    def evaluate_sort(
        self, agents: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # Initialize the domination counts aka fitness
        fitness = np.zeros(len(agents))
        # Loop through all agents
        for i, agent in enumerate(agents):
            if self.problem.breaks_constraint(agent):
                fitness[i] = np.inf
                continue
            # Loop through all other agents
            for j in range(i + 1, len(agents)):
                if self.problem.breaks_constraint(agents[j]):
                    continue
                # Check if the agent is dominated by the other agent
                if self._is_dominated(agent, agents[j]):
                    # If so, increment the domination count of the other agent
                    fitness[i] += 1
                # Check if the other agent is dominated by the agent
                elif self._is_dominated(agents[j], agent):
                    # If so, increment the domination count of the agent
                    fitness[j] += 1

        fitness_idx = fitness.argsort()
        return agents[fitness_idx], fitness[fitness_idx]

    def sort(self, agents: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.evaluate_sort(agents)[0]

    def is_better_than(
        self, agent_a: npt.NDArray[np.float64], agent_b: npt.NDArray[np.float64]
    ) -> bool:
        return (
            self.problem.breaks_constraint(agent_b) or self._dominates(agent_a, agent_b)
        ) and not self.problem.breaks_constraint(agent_a)

    def get_pareto_fronts(self, agents: npt.NDArray[np.float64]) -> List[Set[Any]]:
        """Returns a sorted list of sets of pareto fronts

        The sets themselves are not sorted. The first set is the most dominant
        one, while the last set is dominated by the ones before."""
        remaining_population = set(map(tuple, agents))
        ranking = []
        while remaining_population:
            non_dominated_set = set(
                self._find_non_dominated_agents(remaining_population)
            )
            if len(non_dominated_set) == 0:
                raise RuntimeError(
                    "The set of non dominated individuals can't be empty. This shouldn't happen"
                )
            remaining_population = remaining_population - non_dominated_set
            ranking.append(non_dominated_set)
        return ranking

    def _find_non_dominated_agents(
        self, agents: Iterable[Tuple[Any, ...]]
    ) -> List[npt.NDArray[np.float64]]:
        """Returns a list of all agents that are not dominated by any other agent

        The list is in no specific order. This does not mean that the agents
        necessarily dominate other agents.
        """
        non_dominated_agents = []
        for agent_a in agents:
            is_dominated = any(
                self._dominates(agent_a=agent_b, agent_b=agent_a) for agent_b in agents
            )
            if not is_dominated:
                non_dominated_agents.append(agent_a)

        return non_dominated_agents

    def _dominates(
        self, agent_a: npt.NDArray[np.float64], agent_b: npt.NDArray[np.float64]
    ) -> bool:
        """Returns true if agent_a dominates agent_b with respect to problem

        Definition Pareto-dominating:
        An individual A dominates an individual B iff every objective of A is
        equal or better than those of B and at least one is better
        """
        a_not_worse_than_b = all(
            fun(agent_a) <= fun(agent_b) for fun in self.problem.functions
        )
        a_better_than_b_in_one_objective = any(
            fun(agent_a) < fun(agent_b) for fun in self.problem.functions
        )
        return a_not_worse_than_b and a_better_than_b_in_one_objective

    def _is_dominated(
        self, agent_a: npt.NDArray[np.float64], agent_b: npt.NDArray[np.float64]
    ) -> bool:
        """Return true if agent_a is dominated by agent_b"""
        b_not_worse_than_a = all(
            fun(agent_a) >= fun(agent_b) for fun in self.problem.functions
        )
        b_better_than_a_in_one_objective = any(
            fun(agent_a) > fun(agent_b) for fun in self.problem.functions
        )
        return b_not_worse_than_a and b_better_than_a_in_one_objective

    def __str__(self) -> str:
        return "Non-dominated sorted"
