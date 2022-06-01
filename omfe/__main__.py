from omfe.problems import BinhKorn
from omfe.mga import MicroGeneticAlgorithm
from omfe.evaluator import NonDominatedSortEvaluator
from omfe.visualize import plot_agents

import matplotlib as mpl
import matplotlib.pyplot as plt


def main():
    seed = 2
    problem = BinhKorn()
    evaluator = NonDominatedSortEvaluator(problem)
    mga = MicroGeneticAlgorithm(
        problem=problem,
        evaluator=evaluator,
        population_size=5,
        agents_to_keep=1,
        max_iterations=200,
        num_bits=8,
        seed=seed,
    )
    agents_history = mga.run()

    fig, ax = plt.subplots()  # Convenience method to create figure and plot
    for agents in agents_history:
        plot_agents(axes=ax, agents=agents, problem=mga.problem, marker="o")
        fig.show()
    plt.show(block=True)


if __name__ == "__main__":
    main()
