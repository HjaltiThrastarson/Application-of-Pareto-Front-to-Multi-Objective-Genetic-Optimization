import numpy as np

# Chankong and Haimes Function
def f1(agent):
    return 2 + (agent[0] - 2) ** 2 + (agent[1] - 1) ** 2


def f2(agent):
    return 9 * agent[0] - (agent[1] - 1) ** 2


class MGA:
    def __init__(
        self,
        fitness_functions,
        num_variables,
        population_size=50,
        agents_to_keep=10,
        agents_to_shuffle=8,
        random_restarts=10,
        max_iterations=20,
        num_bits=8,
        random_seed=42,
    ):

        ## User input class variables

        # List of functions which have input of agent with dimensoality of 1xnum_variables
        self.functions = fitness_functions
        # Number of variables i.e dimension of agent
        self.num_variables = num_variables
        # How many random agents to generate
        self.population_size = population_size
        # How many agents to keep after each iteration
        self.agents_to_keep = agents_to_keep
        # How many numbers to shuffle after each iteration
        self.agents_to_shuffle = agents_to_shuffle

        if self.agents_to_keep < self.agents_to_shuffle:
            raise ValueError("agents_to_keep must be smaller than agents_to_shuffle")
        if self.agents_to_keep > self.population_size:
            raise ValueError(
                "Number of agents to keep cannot be greater than population size"
            )

        # How many random restarts to do
        self.random_restarts = random_restarts
        # How many iterations per random restart to do
        self.max_iterations = max_iterations
        # Number of bits to use in graymapping, more --> slower but more accurate
        self.num_bits = num_bits
        # Set random seed for generating agents
        np.random.seed(random_seed)

        ## Internal class variables

        # Current Fitness of all agents and corresponding variables
        self.fitness = np.zeros(population_size)
        # Fitness history, so that we can plot it
        self.fitnessHistory = np.zeros(
            (random_restarts, max_iterations, population_size)
        )
        # Best fitness so far
        self.best_fitness = np.Infinity
        # Initial random start for agents
        self.agents = self.initializeAgents()
        # Agent history
        self.agentsHistory = np.zeros(
            (random_restarts, max_iterations, population_size, num_variables)
        )

    def initializeAgents(self):
        ## Random restart for agents
        agents = np.zeros((self.population_size, self.num_variables))
        for i in range(self.population_size):
            for j in range(self.num_variables):
                # Generates random variables in range -20 to 20
                agents[i][j] = (np.random.rand(1) - 0.5) * 40
        return agents

    def calcFitnessfunction(self, random_restart, iteration):
        ## Calculates fitness for all agents with equal weighting for all functions
        fitness = 0
        for i in range(len(self.agents)):
            for j in range(len(self.functions)):
                # Calculate fitness for each function and add them up
                fitness += self.functions[j](self.agents[i])
            self.fitness[i] = fitness
            if fitness < self.best_fitness:
                self.best_fitness = fitness
            self.fitnessHistory[random_restart][iteration][i] = fitness
            self.agentsHistory[random_restart][iteration][i] = self.agents[i]

    def evaluateAgents(self):
        ## Evalutes fitness of each agent and decides which to keep
        # This is a magic line that sorts the agents by fitness
        agentsSorted = np.array([x for _, x in sorted(zip(self.fitness, self.agents))])
        return agentsSorted[: self.agents_to_keep]

    def shuffleAgents(self, agentsToKeep):
        ## Shuffle agents, generate a random cutoff point and select the best self.number_to_shuffle agents to be trailing agents
        newAgents = np.zeros((self.population_size, self.num_variables))
        agentsToKeepFully = self.agents_to_keep - self.agents_to_shuffle
        newAgents[:agentsToKeepFully] = agentsToKeep[:agentsToKeepFully]
        # Generate random cutoff point
        cutoff = np.random.randint(0, self.num_bits)
        # Set first agent to keep fully to trailing agent
        trailingAgent = 1
        for i in range(agentsToKeepFully, self.agents_to_keep):
            for j in range(self.num_variables):
                # Don't know how to do without for loop, might be a better way
                for h in range(agentsToKeepFully):
                    if trailingAgent == h:
                        if h == trailingAgent:
                            trailingAgent += 1
                            # Map trailing and leading to graycode
                            grayCodeMappingLeading = self.grayMapping(
                                agentsToKeep[i][j]
                            )
                            grayCodeMappingTrailing = self.grayMapping(
                                agentsToKeep[trailingAgent - 1][j]
                            )
                            # Combine them to make a new agent
                            newAgent = (
                                grayCodeMappingLeading[:cutoff]
                                + grayCodeMappingTrailing[cutoff:]
                            )
                            newAgent = self.reverseGrayMapping(int(newAgent, 2))
                            newAgents[i][j] = newAgent
                            break
                        if trailingAgent == agentsToKeepFully:
                            trailingAgent = 1

        # Randomly generate the rest of the agents
        for i in range(self.agents_to_keep, self.population_size):
            for j in range(self.num_variables):
                newAgents[i][j] = (np.random.rand(1) - 0.5) * 40
        self.agents = newAgents

    def grayMapping(self, number):
        ## Graymapping of agents
        # These ranges should be variable specific, can later be implemented in the class
        min_range = -20
        max_range = 20
        total_range = abs(max_range - min_range)
        intervals = total_range / self.num_bits
        # Make sure number is non-negative
        normalizedNumber = number + abs(min_range)
        # Calculate the number of intervals the number is in
        numberOfIntervals = int(normalizedNumber / intervals)
        binaryNumber = format(numberOfIntervals, "0{}b".format(self.num_bits))
        # Convert to greymapping
        n = int(binaryNumber, 2)  # convert to int
        mask = n
        while mask != 0:
            mask >>= 1
            n ^= mask
        return format(n, "0{}b".format(self.num_bits))

    def reverseGrayMapping(self, number):
        ## Does reverse graymapping, takes in a number
        normalizedNumber = number ^ (number >> 1)
        min_range = -20
        max_range = 20
        total_range = abs(max_range - min_range)
        intervals = total_range / self.num_bits
        number = number * intervals
        number = number - abs(min_range)
        return number

    def runIterations(self):
        for random_restart in range(self.random_restarts):
            print(
                "Random restart: {} starting, best fitness is {}".format(
                    random_restart, self.best_fitness
                )
            )
            for iteration in range(self.max_iterations):
                self.calcFitnessfunction(random_restart, iteration)
                agentsToKeep = self.evaluateAgents()
                self.shuffleAgents(agentsToKeep)


def main():
    fitness_functions = [f1, f2]
    mga = MGA(
        fitness_functions,
        num_variables=2,
        population_size=50,
        agents_to_keep=10,
        agents_to_shuffle=5,
        random_restarts=100,
        max_iterations=500,
        num_bits=128,
        random_seed=42,
    )
    mga.runIterations()


if __name__ == "__main__":
    main()
