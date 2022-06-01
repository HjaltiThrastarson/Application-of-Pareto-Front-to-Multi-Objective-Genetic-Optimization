import pytest
from omfe.graymapping import GrayMapper


def test_zero():
    # Setup
    gm = GrayMapper((0, 100), num_bits=8)

    # Execute
    mapped = gm.map(0)

    # Compare
    expected = "00000000"
    assert mapped == expected


def test_map():
    # Setup
    gm = GrayMapper((0, 100), num_bits=32)

    # Execute
    mapped = gm.map(100)

    # Compare
    assert mapped == "10000000000000000000000000000000"


def test_reverse_bit_accurate():
    # Setup
    gm = GrayMapper((0, 255))

    # Execute
    mapped = gm.map(42)
    reverse = gm.reverse_map(mapped)

    # Compare
    assert 42 == reverse


def test_reverse():
    # Setup
    gm = GrayMapper((0, 100))

    # Execute
    mapped = gm.map(42)
    reverse = gm.reverse_map(mapped)

    # Compare
    assert 41.99 < reverse < 42.01


def test_mapping_moved_zero():
    # Setup
    gm = GrayMapper((-10, -5), num_bits=8)

    # Execute
    mapped_low = gm.map(-10)
    mapped_high = gm.map(-5)

    # Compare
    assert "00000000" == mapped_low
    assert "10000000" == mapped_high


def test_exception_num_out_of_range():
    # Setup
    gm = GrayMapper((0, 10), num_bits=32)

    # Execute
    with pytest.raises(ValueError, match=r"out of .* range"):
        gm.map(100)

    with pytest.raises(ValueError, match=r"out of .* range"):
        gm.map(-1)


def test_exception_len_bits_not_matching():
    # Setup
    gm = GrayMapper((0, 10), num_bits=32)

    # Execute
    mapped = gm.map(5)

    with pytest.raises(ValueError, match=r"bits"):
        gm.reverse_map(mapped[:-1])


def test_float_mapping():
    # Setup
    gm = GrayMapper((-20, 20), num_bits=32)

    # Execute
    mapped = gm.map(10.9582)
    reverse = gm.reverse_map(mapped)

    assert 10.95 < reverse < 10.96
