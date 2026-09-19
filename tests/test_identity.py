import functools
import inspect

from regression_records.identity import (
    FunctionIdentity,
    _normalize_source,
    compute_code_hash,
)


def tagged(label):
    """A decorator factory, standing in for ``record_func(tolerance=...)``."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def plain(x):
    return x + 1


@tagged("a")
def decorated(x):
    return x + 1


def annotated(x):  # trailing remark
    # full-line remark
    return x + 1


def marker_first(x):
    return "#" + "first"


def marker_second(x):
    return "#" + "second"


def normalized(func, name="f"):
    """``func``'s normalized source with its own name replaced, to compare bodies."""
    return _normalize_source(inspect.getsource(func)).replace(func.__name__, name)


def test_decorator_lines_are_not_part_of_the_hash():
    """Otherwise editing ``@record_func(tolerance=...)`` reports the function changed."""
    assert normalized(decorated) == normalized(plain)


def test_comments_are_not_part_of_the_hash():
    assert "remark" not in normalized(annotated)
    assert normalized(annotated) == normalized(plain)


def test_a_hash_character_inside_a_string_is_not_a_comment():
    """Cutting each line at ``#`` hid every change that followed one in a string."""
    assert '"first"' in normalized(marker_first)
    assert normalized(marker_first) != normalized(marker_second)
    assert compute_code_hash(marker_first) != compute_code_hash(marker_second)


def test_a_lambda_still_gets_a_hash():
    assert compute_code_hash(lambda x: x) != "unknown"


def test_identity_looks_through_wrappers():
    identity = FunctionIdentity.from_function(decorated)
    assert identity.qualname == "decorated"
    assert identity.path_parts[-1] == "decorated"
