import subprocess
import sys

import pytest

from regression_records._optional import library_is_absent


def test_a_missing_library_counts_as_absent():
    with pytest.raises(ModuleNotFoundError) as caught:
        import a_library_that_is_not_installed  # noqa: F401
    assert library_is_absent(caught.value, "a_library_that_is_not_installed")


def test_a_broken_install_does_not_count_as_absent():
    """Hiding it would record DataFrames through the generic object fallback."""
    missing_dependency = ModuleNotFoundError(
        "No module named 'pyarrow'", name="pyarrow"
    )
    assert not library_is_absent(missing_dependency, "pandas")
    assert not library_is_absent(ImportError("DLL load failed"), "pandas")


_RECORD_WITHOUT = """
import sys

sys.modules[{library!r}] = None  # what `import {library}` finds when it is not installed

import numpy as np

from regression_records import record_func
from regression_records.comparison import default_comparator_registry as comparators
from regression_records.serialization import default_registry as serializers

registered = [type(s).__name__ for s in serializers._entries]
registered += [type(c).__name__ for c in comparators._entries]
assert registered and not any(name.startswith({prefix!r}) for name in registered), registered

recorded = record_func(sys.argv[1])(lambda x: x * 2.0)
recorded(np.arange(3))
assert recorded(np.arange(3)).tolist() == [0.0, 2.0, 4.0]
"""


@pytest.mark.parametrize(
    ("library", "prefix"), [("polars", "Polars"), ("pandas", "Pandas")]
)
def test_the_package_records_numpy_without_an_optional_library(
    tmp_path, library, prefix
):
    """None of the frame or series support may be registered, or imported, without it."""
    script = _RECORD_WITHOUT.format(library=library, prefix=prefix)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
