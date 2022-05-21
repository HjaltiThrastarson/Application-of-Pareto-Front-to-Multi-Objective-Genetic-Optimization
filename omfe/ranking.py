"""Module containing classes/functions to rank agents based on different criteria"""


from typing import Iterable, Sequence, List
from numpy.typing import NDArray
import numpy as np

# TODO: Use numpy instead of python inbuilt sets/lists
# TODO: Test weighted sum sort


class WeightedSumSort:
    """Sort agents with n variables by creating a weighted sum of the individual
    objective functions. Lowest fitness score first."""

    def __init__(self, objectives: Iterable) -> None:
        self.objectives = objectives

    def equal_weighted_sum_sort(
        self, agents: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        weights = np.ones(len(agents)) / len(agents)
        return self.weighted_sum_sort(agents, weights)

    def random_weighted_sum_sort(
        self, agents: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        weights = self.get_random_vec_with_sum_one(len(agents))
        return self.weighted_sum_sort(agents, weights)

    def weighted_sum_sort(
        self, agents: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Sort the list of agents with by a weighted sum fitness score of the
        objective functions. Returns a sorted list, lowest score first

        Agents are just arrays of the function inputs, i.e x1, x2, x3, ...
        """

        agent_scores = np.array(
            [self.calculate_fitness(agent, weights) for agent in agents]
        )
        agent_scores_sorted_idx = agent_scores.argsort()
        return agents[agent_scores_sorted_idx]

    def calculate_fitness(
        self, agent: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> np.float64:
        """Calculates the fitness of an agent. Number of weights must be equal
        to number of objectives/problem functions"""
        objective_scores = np.array([fun(agent) for fun in self.objectives])
        return np.dot(objective_scores, weights)

    @staticmethod
    def get_random_vec_with_sum_one(length: int) -> NDArray[np.float64]:
        """Generate a vector that sums up to 1 with uniform distribution

        This can be imagined as cutting a string at random locations and measuring
        the distance of the resulting pieces, thus the ends 0 and 1 need to be added.

        Note: This is NOT choosing every coordinate randomly and then rescaling
        to [0,1], as this would not result in a uniform distribution. See
        https://stackoverflow.com/a/8068956 for an explanation attempt.
        """
        cuts = np.concatenate([0, np.random.random(size=length - 1), 1], axis=None)
        cuts.sort()
        return np.diff(cuts)


class NonDominatedSort:
    """Sort agents with n variables according to non-dominance of the given
    objective functions

    Objectives is a list
    """

    def __init__(self, objectives: Iterable) -> None:
        self.objectives = objectives

    def get_flat_non_dominated_ranking(self, agents):
        """Returns a flat list sorted by fronts. Order withing fronts is arbitrary"""
        non_dominated_sort = self.non_dominated_sort(agents)
        return [agent for agent_set in non_dominated_sort for agent in agent_set]

    def non_dominated_sort(self, agents):
        """Returns a sorted list of sets of pareto fronts

        The sets themselves are not sorted. The first set is the most dominant
        one, while the last set is dominated by the ones before."""
        remaining_population = set(agents)
        ranking = []
        while remaining_population:
            non_dominated_set = set(
                self.find_non_dominated_agents(remaining_population)
            )
            remaining_population = remaining_population - non_dominated_set
            ranking.append(non_dominated_set)
        return ranking

    def find_non_dominated_agents(self, agents):
        """Returns a list of all agents that are not dominated by any other agent

        The list is in no specific order. This does not mean that the agents
        necessarily dominate other agents.
        """
        non_dominated_agents = []
        for agent_a in agents:
            is_dominated = any(self.dominates(agent_b, agent_a) for agent_b in agents)
            if not is_dominated:
                non_dominated_agents.append(agent_a)

        return non_dominated_agents

    def dominates(self, agent_a, agent_b):
        """Returns true if agent_a dominates agent_b with respect to problem

        An individual A dominates an individual B iff every objective of A is equal or better than those of B and at least on is better
        """
        a_not_worse_than_b = all(
            fun(agent_a) <= fun(agent_b) for fun in self.objectives
        )
        a_better_than_b_in_one_objective = any(
            fun(agent_a) < fun(agent_b) for fun in self.objectives
        )
        return a_not_worse_than_b and a_better_than_b_in_one_objective
