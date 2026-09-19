"""
DataFrames
==========

Dataframes are the case that makes recording worth the trouble. Nobody wants to write
assertions over a hundred columns by hand, and nobody wants a golden CSV either - it
loses dtypes, and it diffs badly.

This page is a side-by-side tour: each section does the same thing in polars and then in
pandas, and says where the two genuinely differ. It needs both extras,
``pip install "regression-records[polars,pandas]"``. Without them the package imports
and records normally, and their serializers are simply not registered.
"""

import sys
import warnings
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

import numpy as np
import pandas as pd
import polars as pl
from _shared.demo import fresh_store, show_record, show_result

from regression_records import (
    RegressionMismatchError,
    SerializationError,
    ToleranceConfig,
    compare,
    record_func,
)

STORE = fresh_store("10_dataframes")
NAN = float("nan")

# %%
# Recording a frame
# -----------------
#
# The frame is written as parquet - dtypes intact, compressed, content-addressed - and
# the record holds a reference plus its shape. First polars:

SPREAD = 0.0


def book_data(n):
    return {
        "symbol": [f"S{i:02d}" for i in range(n)],
        "price": np.linspace(10.0, 20.0, n) + SPREAD,
        "volume": np.arange(n) * 100,
    }


@record_func(STORE)
def polars_book(n):
    return pl.DataFrame(book_data(n))


print(polars_book(5))
show_record(STORE, keys=["output"], select=lambda r: "polars" in str(r["output"]))

# %%
# and pandas, which is the same on disk apart from the file's prefix. The index is
# part of a pandas frame, so it is part of the parquet file too.


@record_func(STORE)
def pandas_book(n):
    return pd.DataFrame(book_data(n)).set_index("symbol")


print(pandas_book(5))
show_record(STORE, keys=["output"], select=lambda r: "pandas" in str(r["output"]))

# %%
# Column-level diffs
# ------------------
#
# A dataframe comparison reports per column, using the ``['name']`` step in the path
# grammar, and names the rows that differ rather than printing two frames at you.

SPREAD = 0.5
for book in (polars_book, pandas_book):
    try:
        book(5)
    except RegressionMismatchError as e:
        print(f"\n{book.__name__}:")
        show_result(e.result)
SPREAD = 0.0

# %%
# Schema first
# ------------
#
# A schema difference short-circuits the value comparison. Once the columns or their
# dtypes have changed, per-row differences are noise - the frame is a different frame, and
# reporting a thousand "row differs" lines would bury the one fact that matters.

before = {"a": [1, 2], "b": [3.0, 4.0]}
after = {"a": [1.0, 2.0], "c": [3.0, 4.0]}
for frame in (pl.DataFrame, pd.DataFrame):
    print(f"\n{frame.__module__.split('.')[0]}:")
    show_result(compare(frame(before), frame(after)))

# %%
# Column order is part of the schema, in both libraries, and ``strict_dtype=False`` does
# not relax it: that setting is about dtypes, and a frame whose columns moved is a
# different frame to anything that reads it by position.

for frame in (pl.DataFrame, pd.DataFrame):
    result = compare(
        frame({"x": [1], "y": [2]}),
        frame({"y": [2], "x": [1]}),
        ToleranceConfig(strict_dtype=False),
    )
    print(f"{frame.__module__.split('.')[0]}: {result.diffs[0].message}")

# %%
# Tolerance on one column
# -----------------------
#
# ``path_overrides`` keys use the same paths the diffs report, so a column that legitimately
# drifts can be loosened while its neighbours stay strict. Identically for both.

expected = {"price": [10.0, 20.0], "qty": [1.0, 2.0]}
actual = {"price": [10.02, 20.03], "qty": [1.0, 2.000001]}
loose_on_price = ToleranceConfig(
    rtol=1e-7, path_overrides={"$.output['price']": ToleranceConfig(rtol=1e-2)}
)

for frame in (pl.DataFrame, pd.DataFrame):
    print(f"\n{frame.__module__.split('.')[0]}, strict everywhere:")
    show_result(compare(frame(expected), frame(actual), root="$.output"))
    print("loose on price only:")
    show_result(
        compare(frame(expected), frame(actual), loose_on_price, root="$.output")
    )

# %%
# Missing values
# --------------
#
# Here the libraries genuinely differ. polars has two things where pandas has one: a
# ``null`` is a missing value and a ``NaN`` is a floating-point number. A column that
# changes from one to the other has changed - ``mean()`` skips a null and is poisoned by
# a NaN - so it is reported, whatever ``equal_nan`` says.

show_result(compare(pl.DataFrame({"x": [1.0, None]}), pl.DataFrame({"x": [1.0, NAN]})))

# %%
# ``equal_nan`` governs NaN against NaN, and nulls in the same place are always equal.
# Where each side is null is compared first, and the other rows in the column's own
# dtype - so an integer column with a null in it is still compared as integers, and
# identifiers too large for a float are told apart.

nan_frame = pl.DataFrame({"x": [1.0, NAN, None]})
print("NaN vs NaN:             ", compare(nan_frame, nan_frame.clone()).passed)
print(
    "NaN vs NaN, equal_nan off:",
    compare(nan_frame, nan_frame.clone(), ToleranceConfig(equal_nan=False)).passed,
)
print(
    "2**60 + 1 vs 2**60:       ",
    compare(
        pl.DataFrame({"id": [2**60 + 1, None]}), pl.DataFrame({"id": [2**60, None]})
    ).passed,
)

# %%
# A pandas float column has only NaN, so ``None`` and ``NaN`` are the same value by the
# time there is a frame to compare, and ``equal_nan`` decides whether two of them match:

pandas_missing = pd.DataFrame({"x": [1.0, None]})
print("None vs NaN:", compare(pandas_missing, pd.DataFrame({"x": [1.0, NAN]})).passed)

# %%
# Nested columns
# --------------
#
# polars columns can hold lists, arrays and structs. Tolerance reaches inside them: list
# lengths must match row by row, float and integer leaves compare within the column's
# tolerance, and every other leaf exactly. Differences are reported by row.

curves = pl.DataFrame(
    {
        "tenors": [[0.5, 1.0, 2.0], [1.0]],
        "quote": [{"mid": 101.25, "source": "A"}, {"mid": 99.5, "source": "B"}],
    }
)
noise = curves.with_columns(pl.col("tenors").list.eval(pl.element() + 1e-13))
print("float noise inside a list:", compare(curves, noise).passed)

moved = curves.with_columns(
    pl.col("quote").struct.with_fields(pl.field("mid") + pl.Series([0.0, 0.25]))
)
show_result(compare(curves, moved))

# %%
# Series
# ------
#
# A Series records as a one-column parquet file that keeps its name, and compares like a
# column: name and dtype, then length (in pandas, the index), then values. An unnamed
# polars series is named ``""``; an unnamed pandas one has no name at all, which parquet
# cannot represent, so the record carries it separately.


@record_func(STORE)
def polars_returns(prices):
    return prices.pct_change().alias("return")


@record_func(STORE)
def pandas_rolling(values):
    series = pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)))
    return series.rolling(2).mean()


print(polars_returns(pl.Series("price", [100.0, 101.0, 99.0])))
print(pandas_rolling([1.0, 2.0, 3.0, 4.0]))
show_record(
    STORE, keys=["invocation", "output"], select=lambda r: "polars.Series" in str(r)
)

# %%
# Wide frames
# -----------
#
# Reporting is capped: once enough columns have differed, the rest are replaced by a
# single summary line. A frame where every column shifted should say so once, not once per
# column.

for frame in (pl.DataFrame, pd.DataFrame):
    result = compare(
        frame({f"c{i}": [0.0] for i in range(40)}),
        frame({f"c{i}": [1.0] for i in range(40)}),
    )
    print(f"{len(result.diffs)} diff(s) for 40 differing columns")
    print(result.diffs[-1].message)

# %%
# What is refused
# ---------------
#
# Each of these raises ``SerializationError`` when recording, rather than writing a
# record that lost something or could never be read back.
#
# * A polars ``LazyFrame`` is not auto-collected. Collection can be expensive or
#   side-effecting, so calling ``.collect()`` stays your decision, made where you can see
#   the cost.
# * A polars ``Object`` column is the one polars dtype parquet cannot hold.
# * A pandas ``object`` column of mixed types is something Arrow cannot represent.

from regression_records.serialization import (
    SerializationContext,
    default_registry,
    to_json_tree,
)

awkward = pd.DataFrame({"mixed": [1, "two", {"three": 3}]})
refused = {
    "polars LazyFrame": pl.LazyFrame({"a": [1]}),
    "polars Object column": pl.DataFrame(
        {"payload": pl.Series([object()], dtype=pl.Object)}
    ),
    "pandas mixed column": awkward,
}

ctx = SerializationContext(data_dir=STORE / "scratch", registry=default_registry)
for label, value in refused.items():
    try:
        to_json_tree(value, ctx)
    except SerializationError as e:
        print(f"{label}: {str(e)[:100]}...")

# %%
# Opting in to pickle
# ^^^^^^^^^^^^^^^^^^^
#
# pandas, and only pandas, has a way out. Unpickling runs code, and a store is committed
# and shared, so a pickle is never written or read unless *you* asked for that: build
# the pandas serializers with ``allow_pickle=True`` and swap them into a copy of the
# default registry. The fallback then applies **to that one file**, with a warning,
# because a pickled record is less portable and not inspectable. polars has no such
# option: every dtype but ``Object`` has a parquet form, so there is always a cast that
# makes the frame recordable.

from regression_records.serialization.pandas_serializer import (
    PandasDataFrameSerializer,
)

pickling = default_registry.copy()
pickling.register(PandasDataFrameSerializer(allow_pickle=True), replace=True)
ctx = SerializationContext(data_dir=STORE / "scratch", registry=pickling)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    tree = to_json_tree(awkward, ctx)

print("stored as:", tree["format"])
for w in caught:
    print(f"{w.category.__name__}: {str(w.message)[:110]}...")

# %%
# Three things to know about frames on disk
# -----------------------------------------
#
# * **Parquet is content-addressed**, so a frame recorded identically by ten different
#   functions is one file per identity directory - and re-recording an unchanged frame
#   writes nothing.
# * **A frame passed as an argument is matched by its parquet bytes.** Those are stable
#   from run to run, but not across every polars or pyarrow release: after an upgrade
#   that changes the encoding, calls taking a frame are seen as new cases rather than
#   validated against the old ones. Re-record after such an upgrade, and see example 13.
# * ``DatetimeIndex.freq`` does not survive parquet. An index round-trips with its
#   values and dtype but not its inferred frequency. If ``freq`` is part of what you are
#   testing, assert on it separately.

print("parquet files:", sorted(p.name for p in STORE.rglob("*.parquet")))
