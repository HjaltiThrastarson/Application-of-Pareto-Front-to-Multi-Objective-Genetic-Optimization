import matplotlib.pyplot as plt
import numpy as np


def main():
    print("Hello World!")
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    plt.show()


if __name__ == "__main__":
    main()
