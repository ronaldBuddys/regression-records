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
