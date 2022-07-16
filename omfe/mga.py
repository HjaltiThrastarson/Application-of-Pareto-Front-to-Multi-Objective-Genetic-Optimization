"""Main module to create, parameterize and run a micro genetic algorithm"""
from typing import Optional
import numpy as np
import numpy.typing as npt

from omfe.problems import Problem
from omfe.graymapping import GrayMapper
from omfe.evaluator import Evaluator

# TODO: Make separate Shuffling class


class AlgorithmRunner:
    """Runs an algorithm the given number of times"""

    def __init__(self, algorithm, seed=42) -> None:
        self.algorithm = algorithm
        self.rng = np.random.default_rng(seed)

    def run(self, times=10) -> npt.NDArray[np.float64]:
        """Run the supplied algorithm `times` times with a different random
        initialization

        Args:
            times (int): The number of times to randomly re-initialize and run
            the algorithm given in the constructor
        """
        best_agent_of_every_random_restart = np.empty(
            shape=(times, self.algorithm.problem.num_variables)
        )
        for i in range(times):
            self.algorithm.clean()
            agents_history = self.algorithm.run()
            best_agent_of_every_random_restart[i] = agents_history[-1][0]

        return best_agent_of_every_random_restart


class MicroGeneticAlgorithm:
    """Implements a microgenetic algorithm configurable through parameters

    The main idea is that this doesn't perform mutation - i.e. the steps
    performed are:
    1. Create a random population of N individuals
    2. Select parents
    3. Generate children using crossover
    4. Check stopping criteria, if not met go back to 2.
    """

    def __init__(
        self,
        problem: Problem,
        evaluator: Evaluator,
        population_size=5,
        agents_to_keep=1,
        max_iterations=20,
        num_bits=32,
        seed=42,
    ) -> None:
        """Initialize a new micro genetic algorithm
        Args:
            problem (Problem): The problem the algorithm should run on. This
            includes the objective functions, the constraints and the search
            domain.
            population_size (int, optional): The total size of the population,
            i.e. the number of individuals/agents. Defaults to 5.
            agents_to_keep (int, optional): The number of best individuals/agents
            that should be copied into the next generation without mutation. Defaults to 1.
            max_iterations (int, optional): Maximum number of iteration before
            the algorithm is terminated. Defaults to 20.
            num_bits (int, optional): Number of bits used for gray coding of
            the agents/individuals. Defaults to 8.
        """
        if population_size < 1:
            raise ValueError("Population must contain at least one individual")

        if agents_to_keep < 0:
            raise ValueError(
                "Number of individuals that are directly copied to the next"
                "generation needs to be 0 or greater"
            )

        if agents_to_keep > population_size:
            raise ValueError(
                "The number of individuals to keep between generations needs"
                "to be smaller than the number of individuals in the population"
            )

        self.problem = problem
        self.evaluator = evaluator
        self.population_size = population_size
        self.agents_to_keep = agents_to_keep
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)
        self.agents_to_shuffle = population_size - agents_to_keep

        if self.agents_to_shuffle % 2:
            raise ValueError(
                "The number of parents must be even. i.e."
                "population_size - agents_to_keep % 2 == True"
            )

        self.gray_mapper = GrayMapper(
            (
                min(val[0] for val in problem.search_domain),
                max(val[1] for val in problem.search_domain),
            ),
            num_bits=num_bits,
        )  # TODO: Separate gray mappers for every variable according to its search domain
        self.agents = self._initialize_agents(only_valid=True)
        self.agents_history = np.zeros(
            shape=(
                self.max_iterations + 1,
                self.population_size,
                self.problem.num_variables,
            ),
            dtype=np.float64,
        )
        self.agents_history[0] = self.agents

    def clean(self, only_valid: Optional[bool] = False) -> None:
        """Reset the current state of the genetic algorithm including the state
        of the evaluator, but keeping the settings (e.g. population size, ...)

        Args:
            only_valid (bool, optional): Whether the new initial population should
            be regenerated until it lies within the constraints. Defaults to False.
        """
        self.evaluator.reset()
        self.agents = self._initialize_agents(only_valid=only_valid)
        self.agents_history = np.zeros(
            shape=(
                self.max_iterations + 1,
                self.population_size,
                self.problem.num_variables,
            ),
            dtype=np.float64,
        )
        # TODO: Instead of assigning the new agents here and on every iteration,
        # make self.agents a "view" into the history at the current index.
        self.agents_history[0] = self.agents

    def _initialize_agent(
        self, only_valid: Optional[bool] = False
    ) -> npt.NDArray[np.float64]:
        """Randomly initializes a single agent. If only_valid is set to true,
        it will retry until no constraints are broken
        """
        ranges = np.array(self.problem.search_domain)
        agent = self.rng.uniform(low=ranges[:, 0], high=ranges[:, 1])

        if not only_valid:
            return agent

        while not self.problem.is_inside_constraints(agent):
            agent = self.rng.uniform(low=ranges[:, 0], high=ranges[:, 1])

        return agent

    def _initialize_agents(
        self, only_valid: Optional[bool] = False
    ) -> npt.NDArray[np.float64]:
        """Create a randomly initialized population

        Returns:
            npt.NDArray[np.float64]: An array of agents/individuals (which are
            arrays of values as well)
        """
        return np.array(
            [self._initialize_agent(only_valid) for _ in range(self.population_size)]
        )

    def _select_parents(self) -> npt.NDArray[np.float64]:
        """Use the evaluator to rank agents and select the best ones for crossover

        Returns:
            npt.NDArray[np.float64]: Returns the selected parents
        """
        agents = self.evaluator.sort(self.agents)
        return agents[: self.agents_to_shuffle]

    def _shuffle_agents(
        self, agents: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        cutoff = self.rng.integers(low=0, high=self.gray_mapper.num_bits)
        parent_idx = self.rng.choice(
            len(agents), size=(len(agents) // 2, 2), replace=False
        )
        children = np.empty(agents.shape)
        for idx1, idx2 in parent_idx:
            children[idx1], children[idx2] = self._crossover_parents(
                np.array([agents[idx1], agents[idx2]]), cutoff
            )
        return children

    def _crossover_parents(
        self, agents: npt.NDArray[np.float64], cutoff: int
    ) -> npt.NDArray[np.float64]:
        parents_gray: npt.NDArray[np.float64] = np.array(
            [self.gray_mapper.map(var) for var in agents.flat]
        ).reshape(agents.shape)
        parent0 = "".join(parents_gray[0])
        parent1 = "".join(parents_gray[1])
        child0 = parent0[:cutoff] + parent1[cutoff:]
        child1 = parent1[:cutoff] + parent0[cutoff:]
        child0_gray = [
            child0[i : i + self.gray_mapper.num_bits]
            for i in range(0, len(child0), self.gray_mapper.num_bits)
        ]
        child1_gray = [
            child1[i : i + self.gray_mapper.num_bits]
            for i in range(0, len(child1), self.gray_mapper.num_bits)
        ]
        children = np.array(
            [
                self.gray_mapper.reverse_map(var)
                for var in np.array([child0_gray, child1_gray]).flat
            ]
        ).reshape(parents_gray.shape)
        return children

    def _crossover_parents_every_var(
        self, agents: npt.NDArray[np.float64], cutoff: int
    ) -> npt.NDArray[np.float64]:
        """Crosses two agents at the given cutoff point and returns the crossed
        children

        Maps the parents to graycode before crossover. Every variable of the
        parent agent is cut at the same point and crossed with the corresponding
        variable of the other parent. e.g.:

        agent1: aaaa|aa, bbbb|bb, cccc|cc
        agent2: dddd|dd, eeee|ee, ffff|ff

        =>

        child1: aaaa|dd, bbbb|ee, cccc|ff
        child2: dddd|aa, eeee|bb, ffff|cc

        Then the graymapping of every variable is reversed before the children
        are returned.

        Args:
            agents (npt.NDArray[np.float64]): Two agents that will be crossed

        Returns:
            npt.NDArray[np.float64]: Two children that are made up from the parents
        """
        parents_gray = np.array(
            [self.gray_mapper.map(var) for var in agents.flat]
        ).reshape(agents.shape)
        for var in parents_gray.T:
            tmp = var[0]
            var[0] = var[0][:cutoff] + var[1][cutoff:]
            var[1] = var[1][:cutoff] + tmp[cutoff:]
        children = np.array(
            [self.gray_mapper.reverse_map(var) for var in parents_gray.flat]
        ).reshape(parents_gray.shape)
        return children

    def run(self) -> npt.NDArray[np.float64]:
        """Run the micro genetic algorithm

        Returns:
            npt.NDArray[np.float64]: A numpy 3D array with a list of all agents of
            every generation
        """
        for itr in range(self.max_iterations):
            parents = self._select_parents()
            children = self._shuffle_agents(parents)
            new_generation = np.concatenate(
                (parents[: self.agents_to_keep], children), axis=0
            )
            self.agents = new_generation
            self.agents_history[itr + 1] = new_generation  # 0 is the start generation
        return self.agents_history
