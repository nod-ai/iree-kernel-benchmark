from .utils import match_lines
import pytest

# Tests for the basic functionality of match_lines().


def test_match_nothing():
    match_lines("The content of this text is irrelevant.", [])


def test_successful_matches():
    match_lines(
        """
The quick brown fox
jumps over
the lazy dog.
""",
        ["quick brown", "jumps over", "dog."],
    )


@pytest.mark.xfail(strict=True)
def test_partial_match():
    match_lines(
        """
The quick brown fox
jumps over
the lazy dog.
""",
        ["quick brown", "jumps over", "cat."],
    )


@pytest.mark.xfail(strict=True)
def test_no_match():
    match_lines(
        """
The quick brown fox
jumps over
the lazy dog.
""",
        ["slow purple", "walks past", "cat."],
    )
