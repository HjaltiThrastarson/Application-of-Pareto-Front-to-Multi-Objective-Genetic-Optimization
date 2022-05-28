import numpy as np
import matplotlib.pyplot as plt
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video
from benchmark_chankong_haimes import ChankongHaimes as CH
from pymoo.problems.multi.bnh import BNH
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.optimize import minimize

"""Plot Functions only plot, Rec functions will produce a recording showing each iteration of a given random restart.
By default, only the agents_to_keep will be recorded. Call function with ShowRandom=True to show the randomly
generated parameters (or their fitness respectively)"""

"""To plot benchmarks, call the function parameter "benchmark" string 'CH' or 'BNH'"""

def VarPlot(mga, benchmark='False'):


    if benchmark == 'CH':
        problem = CH()
        algorithm = NSGA2()
        termination = MultiObjectiveDefaultTermination()
        result = minimize(
            problem,
            algorithm,
            termination,
            pf=True,
            seed=1,
            save_history=True,
            verbose=True,
        )

    if benchmark == 'BNH':
        problem = BNH()
        algorithm = NSGA2()
        termination = MultiObjectiveDefaultTermination()
        result = minimize(
            problem,
            algorithm,
            termination,
            pf=True,
            seed=1,
            save_history=True,
            verbose=True,
        )


    [UniqueAgents, indices] = np.unique(mga.best_agents, axis=0, return_index=True)
    for entry in enumerate(sorted(indices)):
        plt.scatter(mga.best_agents[0:entry[1]+1, 0], mga.best_agents[0:entry[1]+1, 1], facecolors='none', edgecolors='b')
        if benchmark != 'False':
            plt.scatter(result.X[:, 0], result.X[:, 1], facecolors='none', edgecolors='r', alpha=1)
        plt.title('Optimal parameter choices')
        plt.xlabel('x')
        plt.ylabel('y')
        if entry[0] == len(indices)-1:
            plt.show()



def FitPlot(mga, benchmark='False'):


    if benchmark == 'CH':
        benchproblem = CH()
        algorithm = NSGA2()
        termination = MultiObjectiveDefaultTermination()
        result = minimize(
            benchproblem,
            algorithm,
            termination,
            pf=True,
            seed=1,
            save_history=True,
            verbose=True,
        )

    if benchmark == 'BNH':
        benchproblem = BNH()
        algorithm = NSGA2()
        termination = MultiObjectiveDefaultTermination()
        result = minimize(
            benchproblem,
            algorithm,
            termination,
            pf=True,
            seed=1,
            save_history=True,
            verbose=True,
        )


    f1 = np.zeros(len(mga.best_agents))
    f2 = np.zeros(len(mga.best_agents))
    for i in range(len(mga.best_agents)):
        f1[i] = mga.problem.f_1(mga.best_agents[i])
        f2[i] = mga.problem.f_2(mga.best_agents[i])
        plt.scatter(f1, f2, facecolors='none', edgecolors='b')
        if benchmark != 'False':
            plt.scatter(result.F[:, 0], result.F[:, 1], facecolors='none', edgecolors='r', alpha=1)
        plt.title('Fitness Scatter')
        plt.xlabel('f1')
        plt.ylabel('f2')
        if i == len(mga.best_agents)-1:
            plt.show()


def FitHistRec(mga, randomrestart, benchmark='False', ShowRandom=False):


    if benchmark == 'CH':
        problem = CH()
        algorithm = NSGA2()
        termination = MultiObjectiveDefaultTermination()
        result = minimize(
            problem,
            algorithm,
            termination,
            pf=True,
            seed=1,
            save_history=True,
            verbose=True,
        )

    if benchmark == 'BNH':
        problem = BNH()
        algorithm = NSGA2()
        termination = MultiObjectiveDefaultTermination()
        result = minimize(
            problem,
            algorithm,
            termination,
            pf=True,
            seed=1,
            save_history=True,
            verbose=True,
        )


    with Recorder(Video("FitIter.mp4")) as rec:

        for i in range(mga.max_iterations):
            if ShowRandom:
                for j in range(mga.population_size):
                    if j < 10:
                        plt.scatter(mga.problem.f_1(mga.agents_history[randomrestart-1][i][j]),
                                    mga.problem.f_2(mga.agents_history[randomrestart-1][i][j]), facecolors='none',edgecolors='b')
                    else:
                        plt.scatter(mga.problem.f_1(mga.agents_history[randomrestart-1][i][j]),
                                    mga.problem.f_2(mga.agents_history[randomrestart-1][i][j]), facecolors='none',edgecolors='g')
                if benchmark != 'False':
                    plt.scatter(result.F[:, 0], result.F[:, 1], facecolors='none', edgecolors='r', alpha=1)
                plt.title(f'Fitness Generation {i}')
                plt.xlabel('f1')
                plt.ylabel('f2')
                rec.record()
            else:
                for j in range(mga.agents_to_keep):
                        plt.scatter(mga.problem.f_1(mga.agents_history[randomrestart-1][i][j]),
                                    mga.problem.f_2(mga.agents_history[randomrestart-1][i][j]), facecolors='none',edgecolors='b')
                if benchmark != 'False':
                    plt.scatter(result.F[:, 0], result.F[:, 1], facecolors='none', edgecolors='r', alpha=1)
                plt.title(f'Fitness Generation {i}')
                plt.xlabel('f1')
                plt.ylabel('f2')
                rec.record()

def VarHistRec(mga, randomrestart, benchmark='False', ShowRandom=False):


    if benchmark == 'CH':
        problem = CH()
        algorithm = NSGA2()
        termination = MultiObjectiveDefaultTermination()
        result = minimize(
            problem,
            algorithm,
            termination,
            pf=True,
            seed=1,
            save_history=True,
            verbose=True,
        )

    if benchmark == 'BNH':
        problem = BNH()
        algorithm = NSGA2()
        termination = MultiObjectiveDefaultTermination()
        result = minimize(
            problem,
            algorithm,
            termination,
            pf=True,
            seed=1,
            save_history=True,
            verbose=True,
        )

    with Recorder(Video("VarIter.mp4")) as rec:

        for i in range(mga.max_iterations):
            if ShowRandom:
                for j in range(mga.population_size):
                    if j < 10:
                        plt.scatter(mga.agents_history[randomrestart - 1][i][j, 0],
                                    mga.agents_history[randomrestart - 1][i][j, 1], facecolors='none', edgecolors='b')
                    else:
                        plt.scatter(mga.agents_history[randomrestart - 1][i][j, 0],
                                    mga.agents_history[randomrestart - 1][i][j, 1], facecolors='none', edgecolors='g')
                if benchmark != 'False':
                    plt.scatter(result.X[:, 0], result.X[:, 1], facecolors='none', edgecolors='r', alpha=1)
                plt.title(f'Parameter Generation {i}')
                plt.xlabel('x')
                plt.ylabel('y')
                rec.record()
            else:
                for j in range(mga.agents_to_keep):
                    plt.scatter(mga.agents_history[randomrestart - 1][i][j, 0],
                                mga.agents_history[randomrestart-1][i][j,1], facecolors='none', edgecolors='b')
                if benchmark != 'False':
                    plt.scatter(result.X[:, 0], result.X[:, 1], facecolors='none', edgecolors='r', alpha=1)
                plt.title(f'Fitness Generation {i}')
                plt.xlabel('f1')
                plt.ylabel('f2')
                rec.record()