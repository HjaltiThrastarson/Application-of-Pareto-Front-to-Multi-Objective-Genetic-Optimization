from omfe.evaluator import NonDominatedSortEvaluator
from omfe.problems import BinhKorn
import numpy as np

# TODO: Test weighted sum sort


def test_dominance_ranking_simple():
    # Setup
    problem = BinhKorn()
    ranking_algo = NonDominatedSortEvaluator(problem)
    agents = np.array([(1, 3), (3, -1), (2, 2)])

    # Execute
    ranking = ranking_algo.evaluate_sort(agents)

    # Compare
    expected_ranking = np.array([(2, 2), (1, 3), (3, -1)])
    assert np.all(ranking[0] == expected_ranking)


def test_dominance_ranking_constraints():
    problem = BinhKorn()
    ranking_algo = NonDominatedSortEvaluator(problem)
    agents = np.array(
        [
            (-1, -1),
            (1, -1),
        ]
    )

    # Execute
    ranking = ranking_algo.evaluate_sort(agents)

    # Compare
    expected_ranking = np.array([(1, -1), (-1, -1)])
    assert np.all(ranking[0] == expected_ranking)
