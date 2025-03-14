from collections import deque
import pytest


def match_lines(text, substrings):
    """
    Takes a whole text (with newlines) and an iterable of substrings to match.
    Assumes each substring is found on its own line and tries to match each
    substring in order. If a substring can't be found, makes the test fail with
    a human-readable error.

    Intended to serve a similar function to FileCheck's CHECK lines.
    """

    text_as_lines = text.splitlines()
    # Cursor used to ensure lines are matched only once and in order, and to
    # provide feedback when a match fails. Lines before the cursor are excluded
    # from future searches.
    lines_cursor = 0

    for substrings_cursor, substring in enumerate(substrings):
        for i in range(lines_cursor, len(text_as_lines)):
            if substring in text_as_lines[i]:
                lines_cursor = i + 1
                break
        else:
            preceding_text = "\n".join(text_as_lines[:lines_cursor])
            remaining_text = "\n".join(text_as_lines[lines_cursor:])
            format_substring = lambda s: "- " + repr(s)
            preceding_matches = "\n".join(
                map(format_substring, substrings[:substrings_cursor])
            )
            remaining_matches = "\n".join(
                map(format_substring, substrings[substrings_cursor:])
            )

            pytest.fail(
                f"""
Couldn't find a line containing {substring!r} in remaining text:

***
{remaining_text}
***

Preceding text (already matched):

***
{preceding_text}
***

Substrings already matched:

{preceding_matches}

Substrings not yet matched:

{remaining_matches}
""",
                pytrace=False,
            )
