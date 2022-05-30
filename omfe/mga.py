# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=consider-using-enumerate


from multiprocessing.sharedctypes import Value
import numpy as np
from numpy.typing import NDArray
from omfe.problems import Problem
from omfe.ranking import Sorter
from omfe.graymapping import GrayMapper


class AlgorithmRunner:
    def __init__(self, algorithm, random_seed=42) -> None:
        self.algorithm = algorithm
        self.rng = np.random.default_rng(seed=random_seed)

    def run(self, times=10):
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
    def __init__(
        self,
        problem: Problem,
        sorter: Sorter,
        population_size=5,
        agents_to_keep=1,
        max_iterations=20,
        num_bits=32,
        seed=42,
    ):
        """Implements a microgenetic algorithm configurable through parameters

        The main idea is that this doesn't perform mutation - i.e. the steps
        performed are:
        1. Create a random population of N individuals
        2. Select parents
        3. Generate children using crossover
        4. Check stopping criteria, if not met go back to 2.

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

        Raises:
            ValueError: _description_
            ValueError: _description_
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
        self.sorter = sorter
        self.population_size = population_size
        self.agents_to_keep = agents_to_keep
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)
        self.agents_to_shuffle = (
            population_size - agents_to_keep
        )  # TODO: Verify that this is actually what is intended. Maybe remove this and calculate when needed, was 8

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
        self.agents = None
        self.agents_history = None
        self.clean(only_valid=True)

    def clean(self, only_valid=False):
        # TODO: Also reinitialize weights of weighted sum?
        self.agents = self.initialize_agents(only_valid=only_valid)
        self.agents_history = np.zeros(
            shape=(
                self.max_iterations + 1,
                self.population_size,
                self.problem.num_variables,
            ),
            dtype=np.float64,
        )
        # TODO: Instead of assigning the new agents here and on every iteration, make self.agents a "view" into the history at the current index.
        self.agents_history[0] = self.agents

    def initialize_agent(self, only_valid=False):
        """Randomly initializes a single agent. If only_valid is set to true,
        it will retry until no constraints are broken
        """
        ranges = np.array(self.problem.search_domain)
        agent = self.rng.uniform(low=ranges[:, 0], high=ranges[:, 1])

        if not only_valid:
            return agent

        while not np.all(self.problem.evaluate_constraints(agent)):
            agent = self.rng.uniform(low=ranges[:, 0], high=ranges[:, 1])

        return agent

    def initialize_agents(self, only_valid=False) -> NDArray[np.float64]:
        """Create a randomly initialized population

        Returns:
            NDArray[np.float64]: An array of agents/individuals (which are
            arrays of values as well)
        """
        return np.array(
            [self.initialize_agent(only_valid) for _ in range(self.population_size)]
        )

    def select_parents(self) -> NDArray[np.float64]:
        """Use the sorter to rank agents and select the best ones for crossover

        Returns:
            NDArray[np.float64]: Returns the selected parents
        """
        # TODO: incorporate constraints somehow. (weighted_sum: add 1000000 to fitness score, non_dominated_sort: ??)
        agents = self.sorter.sort(self.agents)
        return agents[: self.agents_to_shuffle]

    def shuffle_agents(self, agents: NDArray[np.float64]) -> NDArray[np.float64]:
        cutoff = self.rng.integers(low=0, high=self.gray_mapper.num_bits)
        parent_idx = self.rng.choice(
            len(agents), size=(len(agents) // 2, 2), replace=False
        )
        children = np.empty(agents.shape)
        for idx1, idx2 in parent_idx:
            children[idx1], children[idx2] = self.crossover_parents(
                np.array([agents[idx1], agents[idx2]]), cutoff
            )
        return children

    def crossover_parents(
        self, agents: NDArray[np.float64], cutoff: int
    ) -> NDArray[np.float64]:
        parents_gray = np.array(
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

    def crossover_parents_every_var(
        self, agents: NDArray[np.float64], cutoff: int
    ) -> NDArray[np.float64]:
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
            agents (NDArray[np.float64]): Two agents that will be crossed

        Returns:
            NDArray[np.float64]: Two children that are made up from the parents
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

    def run(self) -> NDArray[np.float64]:
        for itr in range(self.max_iterations):
            parents = self.select_parents()
            children = self.shuffle_agents(parents)
            new_generation = np.concatenate(
                (parents[: self.agents_to_keep], children), axis=0
            )
            self.agents = new_generation
            self.agents_history[itr + 1] = new_generation  # 0 is the start generation
        return self.agents_history
