"""Module containing classes/functions to rank agents based on different criteria"""


from typing import Iterable


class NonDominatedSort:
    """Sort agents with n variables according to non-dominance of the given
    objective functions

    Objectives is a list
    """

    def __init__(self, objectives: Iterable) -> None:
        self.objectives = objectives

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
