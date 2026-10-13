"""Tests for ``everalgo.utils.text``."""

import pytest

from everalgo.utils import ascii_width, decode_bytes


@pytest.mark.parametrize(("text", "width"), [("abc", 3), ("中文", 4), ("中a文b", 6), ("", 0)])
def test_ascii_width_counts_east_asian_wide_characters_as_two(text: str, width: int) -> None:
    assert ascii_width(text) == width


@pytest.mark.parametrize("encoding", ["utf-8", "gbk"])
def test_decode_bytes_reads_utf8_and_gbk(encoding: str) -> None:
    assert decode_bytes("编码 test".encode(encoding)) == "编码 test"


def test_decode_bytes_falls_back_to_latin1_so_it_never_raises() -> None:
    undecodable = bytes([0x81, 0xFF, 0xFE])  # neither UTF-8 nor GBK
    assert decode_bytes(undecodable) == undecodable.decode("latin-1")
