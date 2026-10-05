"""Checks that human facing strings do not contain dash characters."""

FORBIDDEN = ("-", "\u2013", "\u2014")


def assert_no_dash(text: str, where: str) -> None:
    for mark in FORBIDDEN:
        if mark in text:
            raise ValueError(f"{where} contains a dash character")
