"""Tests for ``everalgo.utils.json``."""

import json
from datetime import datetime

from everalgo.utils import json_default


def test_a_datetime_becomes_iso_8601() -> None:
    assert json_default(datetime(2025, 3, 1, 12, 0, 0)) == "2025-03-01T12:00:00"


def test_anything_else_becomes_its_str() -> None:
    assert json_default({1, 2, 3}) == "{1, 2, 3}"
    assert json_default(b"hello") == "b'hello'"


def test_it_serves_as_the_json_dumps_default() -> None:
    assert json.dumps({"at": datetime(2025, 3, 1)}, default=json_default) == '{"at": "2025-03-01T00:00:00"}'
