import numpy as np


#Chankong and Haimes Function
def f1(agent):
        return 2 + (agent[0]-2)**2 + (agent[1]-1)**2

def f2(agent):
    return 9*agent[0] - (agent[1]-1)**2



class MGA():
    def __init__(self, fitness_functions, population_size, max_iterations, num_variables):
        self.fitness = np.zeros(population_size)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.num_variables = num_variables
        self.fitnesshistory = np.zeros((population_size, max_iterations))
        self.agents = self.initializeagents(num_variables)
        self.functions = fitness_functions
        self.currentiteration = 0

    def initializeagents(self, num_variables):
        agents = np.zeros((self.population_size, num_variables))
        for i in range(self.population_size):
            for j in range(num_variables):
                agents[i][j] = (np.random.rand(1)-0.5)*40
        return agents

    def calc_fitnessfunc(self):
        fitness = 0
        for i in range (len(self.agents)):
            for j in range(len(self.functions)):
                fitness += self.functions[j](self.agents[i])
            self.fitness[i] = fitness
            self.fitnesshistory[i][self.currentiteration] = fitness

    def evaluate(self):
        pass


    def shuffleagents(self):
        pass

    def graymapping(self):
        min_range = -20
        max_range = 20

def main():
    fitness_functions = [f1, f2]
    population_size = 50
    max_iterations = 100
    num_variables = 2
    mga = MGA(fitness_functions, population_size, max_iterations, num_variables)
    mga.calc_fitnessfunc()
    print(mga.fitness)


if __name__ == "__main__":
    main()