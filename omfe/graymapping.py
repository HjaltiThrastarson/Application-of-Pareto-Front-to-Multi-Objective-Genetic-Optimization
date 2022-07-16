from typing import Tuple
import numpy as np


class GrayMapper:
    def __init__(self, var_range: Tuple[int, int], num_bits=32) -> None:
        self.num_bits = num_bits
        # These ranges should be variable specific, can later be implemented in the class
        self.min_val = var_range[0]
        self.max_val = var_range[1]
        self.total_range = abs(self.max_val - self.min_val)
        self.intervals: float = self.total_range / (2**self.num_bits - 1)

    def map(self, number: np.float64) -> str:
        """Converts `number` to the corresponing graycode with num_bits"""

        if number > self.max_val or number < self.min_val:
            raise ValueError("Supplied number is out of conversion range")

        # Make sure number is non-negative
        normalized_number = number + np.abs(self.min_val)
        # Calculate the number of intervals the number is in
        number_of_intervals = int(normalized_number / self.intervals)
        # Convert to graycode
        binary_number = number_of_intervals ^ (number_of_intervals >> 1)

        return format(binary_number, f"0{self.num_bits}b")

    def reverse_map(self, number_string: str) -> float:
        """Maps a string of bits in graycode mapping back to the original number
        in the conversion range.
        """

        if len(number_string) != self.num_bits:
            raise ValueError("Number of bits doesn't match used graycode bitrange")

        # Takes in gray code decimal equailant number and converts to correct binary number equivalent
        # Taking xor until
        # n becomes zero
        number = int(number_string, 2)

        inv = 0
        while number:
            inv = inv ^ number
            number = number >> 1
        number = inv
        number = number * self.intervals  # type: ignore[assignment]
        number = number - abs(self.min_val)
        assert self.min_val <= number <= self.max_val
        return number
