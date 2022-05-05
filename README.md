# Application of Pareto Front to Multi-Objective Genetic Optimization

## IDEAS

1. Implement micro genetic algorithm
2. Use Chankong and Haimes test func for evaluation
3. The range is [-20, 20], idea is to map small ranges in that range to binary numbers, for example with 1 bit we would have 0 as -20 and 1 as 20

![mga](mga.png)

## Documentation

Write down any decisions you make, any problems you face and any interesting insights you come across.

* Decided for Python as programming language (due to versatility)
* Write from scratch for better learning
* Compare/Benchmark our solution with existing implementations
* First get minimum example working inkl. visualization:
    * mGA from lecture
    * Changkong/Haimes test function

## Literature
* [Multi-Objective Optimization (Wiki)](https://en.wikipedia.org/wiki/Multi-objective_optimization)
* [Pareto-Efficiency (Wiki)](https://en.wikipedia.org/wiki/Pareto_efficiency)
* [A tutorial on multiobjective optimization: fundamentals and evolutionary methods](https://link.springer.com/article/10.1007/s11047-018-9685-y)
* [Multi-objective Optimisation Using Evolutionary Algorithms: An Introduction](https://link.springer.com/chapter/10.1007/978-0-85729-652-8_1)
* [Adaptive weighted sum method for multiobjective optimization: a new method for Pareto front generation](https://link.springer.com/article/10.1007/s00158-005-0557-6)
* [The weighted sum method for multi-objective optimization: new insights](https://link.springer.com/article/10.1007/s00158-009-0460-7)

## Frameworks/Libraries/Resources
* [scikit-opt](https://github.com/guofei9987/scikit-opt)
* [pymoo](https://github.com/anyoptimization/pymoo) - [Corresponding Paper](https://ieeexplore.ieee.org/abstract/document/9078759)
* [Test functions for optimizations (Wikipedia)](https://en.wikipedia.org/wiki/Test_functions_for_optimization)