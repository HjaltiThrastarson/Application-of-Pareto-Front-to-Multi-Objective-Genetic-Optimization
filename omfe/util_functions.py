from numpy.typing import NDArray
import numpy as np


def gray_mapping(number, num_bits, range):
    """
    Returns a graycode equivalent of the number in num_bits
    """
    # These ranges should be variable specific, can later be implemented in the class
    min_range = range[0]
    max_range = range[1]
    total_range = abs(max_range - min_range)
    intervals = total_range / (2**num_bits - 1)
    # Make sure number is non-negative
    normalizedNumber = number + abs(min_range)
    # Calculate the number of intervals the number is in
    numberOfIntervals = int(normalizedNumber / intervals)
    # Convert to graycode
    binaryNumber = numberOfIntervals ^ (numberOfIntervals >> 1)
    return format(binaryNumber, "0{}b".format(num_bits))


def reversegray_mapping(number, num_bits, range):
    """
    Takes in a the number from the graycode mapping i.e int(nr, 2)
    and returns the corresponding number in the original range
    """
    # Takes in gray code decimal equailant number and converts to correct binary number equivalent
    # Taking xor until
    # n becomes zero
    inv = 0
    while number:
        inv = inv ^ number
        number = number >> 1
    number = inv
    min_range = range[0]
    max_range = range[1]
    total_range = abs(max_range - min_range)
    intervals = total_range / (2**num_bits - 1)
    number = number * intervals
    number = number - abs(min_range)
    return number


def get_random_vec_with_sum_one(length: int) -> NDArray[np.float64]:
    """Generate a vector that sums up to 1 with uniform distribution

    This can be imagined as cutting a string at random locations and measuring
    the distance of the resulting pieces, thus the ends 0 and 1 need to be added.

    Note: This is NOT choosing every coordinate randomly and then rescaling
    to [0,1], as this would not result in a uniform distribution. See
    https://stackoverflow.com/a/8068956 for an explanation attempt.
    """
    cuts = np.concatenate([0, np.random.random(size=length - 1), 1], axis=None)
    cuts.sort()
    return np.diff(cuts)
