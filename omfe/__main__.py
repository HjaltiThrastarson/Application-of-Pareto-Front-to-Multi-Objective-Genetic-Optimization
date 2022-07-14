import omfe
import matplotlib as mpl
import matplotlib.pyplot as plt


def main():
    simple_plot()
    tournament()


def simple_plot():
    seed = 2
    problem = omfe.problems.BinhKorn()
    evaluator = omfe.evaluator.NonDominatedSortEvaluator(problem)
    mga = omfe.mga.MicroGeneticAlgorithm(
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
        omfe.benchmark.plot_agents(
            axes=ax, agents=agents, problem=mga.problem, marker="o"
        )
        fig.show()
    plt.show(block=True)


def tournament():
    problem = omfe.problems.ChankongHaimes()
    evaluator = omfe.evaluator.WeightBasedEvaluator(problem)
    tournament_evaluator = omfe.evaluator.NonDominatedSortEvaluator(problem)
    MGA = omfe.mga.MicroGeneticAlgorithm(
        problem,
        evaluator,
        population_size=10,
        agents_to_keep=5,
        agents_to_shuffle=4,
        random_restarts=1000,
        max_iterations=1000,
        iteration_tolerance=10,
        num_bits=64,
        random_seed=0,
    )
    TMS = omfe.tournament_selector.TournamentSelector(
        MGA,
        tournament_evaluator,
        tournament_size=20,
        num_tournaments=100,
        num_iterations=20,
    )
    TMS.run_iterations()


if __name__ == "__main__":
    main()
