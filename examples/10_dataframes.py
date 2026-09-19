# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 10. DataFrames
#
# Dataframes are the case that makes recording worth the trouble. Nobody wants to write
# assertions over a hundred columns by hand, and nobody wants a golden CSV either - it
# loses dtypes, and it diffs badly.
#
# polars and pandas are optional extras: `pip install "regression-records[polars,pandas]"`.
# Without them the package imports and records normally, and their serializers are simply
# not registered.

# %%
import sys
import warnings
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)

import numpy as np
from _shared.demo import fresh_store, show_record, show_result

from regression_records import (
    RegressionMismatchError,
    SerializationError,
    ToleranceConfig,
    compare,
    record_func,
)

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

print(f"polars: {HAS_POLARS}  pandas: {HAS_PANDAS}")
STORE = fresh_store("10_dataframes")

# %% [markdown]
# ## Recording a polars frame
#
# The frame is written as parquet - dtypes intact, compressed, content-addressed - and
# the record holds a reference plus its shape.

# %%
if HAS_POLARS:
    SPREAD = 0.0

    @record_func(STORE)
    def price_book(n):
        return pl.DataFrame(
            {
                "symbol": [f"S{i:02d}" for i in range(n)],
                "price": np.linspace(10.0, 20.0, n) + SPREAD,
                "volume": np.arange(n) * 100,
            }
        )

    print(price_book(5))
    show_record(STORE, keys=["output"])

# %% [markdown]
# ## Column-level diffs
#
# A dataframe comparison reports per column, using the `['name']` step in the path
# grammar, and names the rows that differ rather than printing two frames at you.

# %%
if HAS_POLARS:
    SPREAD = 0.5
    try:
        price_book(5)
    except RegressionMismatchError as e:
        show_result(e.result)
    SPREAD = 0.0

# %% [markdown]
# ## Schema first
#
# A schema difference short-circuits the value comparison. Once the columns or their
# dtypes have changed, per-row differences are noise - the frame is a different frame, and
# reporting a thousand "row differs" lines would bury the one fact that matters.

# %%
if HAS_POLARS:
    show_result(
        compare(
            pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}),
            pl.DataFrame({"a": [1.0, 2.0], "c": [3.0, 4.0]}),
        )
    )

# %% [markdown]
# ## Tolerance on one column
#
# `path_overrides` keys use the same paths the diffs report, so a column that legitimately
# drifts can be loosened while its neighbours stay strict.

# %%
if HAS_POLARS:
    expected = pl.DataFrame({"price": [10.0, 20.0], "qty": [1.0, 2.0]})
    actual = pl.DataFrame({"price": [10.02, 20.03], "qty": [1.0, 2.000001]})

    print("strict everywhere:")
    show_result(compare(expected, actual, root="$.output"))

    print("\nloose on price only:")
    show_result(
        compare(
            expected,
            actual,
            ToleranceConfig(
                rtol=1e-7,
                path_overrides={"$.output['price']": ToleranceConfig(rtol=1e-2)},
            ),
            root="$.output",
        )
    )

# %% [markdown]
# ## Wide frames
#
# Reporting is capped: once enough columns have differed, the rest are replaced by a
# single summary line. A frame where every column shifted should say so once, not once per
# column.

# %%
if HAS_POLARS:
    wide_expected = pl.DataFrame({f"c{i}": [0.0] for i in range(40)})
    wide_actual = pl.DataFrame({f"c{i}": [1.0] for i in range(40)})
    result = compare(wide_expected, wide_actual)
    print(f"{len(result.diffs)} diff(s) for 40 differing columns")
    print(result.diffs[-1].message)

# %% [markdown]
# ## LazyFrame is refused
#
# Not auto-collected. Collection can be expensive or side-effecting, so calling
# `.collect()` stays your decision, made where you can see the cost.

# %%
if HAS_POLARS:
    from regression_records.serialization import (
        SerializationContext,
        default_registry,
        to_json_tree,
    )

    ctx = SerializationContext(data_dir=STORE / "scratch", registry=default_registry)
    try:
        to_json_tree(pl.LazyFrame({"a": [1]}), ctx)
    except SerializationError as e:
        print(f"SerializationError: {e}")

# %% [markdown]
# ## pandas
#
# DataFrame and Series both record, also as parquet. A Series keeps its name - including
# when it has none, which parquet cannot represent directly.

# %%
if HAS_PANDAS:

    @record_func(STORE)
    def resample(values):
        series = pd.Series(
            values, index=pd.date_range("2024-01-01", periods=len(values))
        )
        return series.rolling(2).mean()

    print(resample([1.0, 2.0, 3.0, 4.0]))
    show_record(
        STORE, keys=["output"], select=lambda r: "pandas" in str(r.get("output"))
    )

# %% [markdown]
# ### Frames parquet cannot hold, and opting in to pickle
#
# A column Arrow cannot represent - mixed types in one object column - cannot be written
# as parquet, and by default recording it is an error that says so. Unpickling runs
# code, and a store is committed and shared, so a pickle is never written or read unless
# *you* asked for that: build the pandas serializers with `allow_pickle=True` and swap
# them into a copy of the default registry. The fallback then applies **to that one
# file**, with a warning, because a pickled record is less portable and not inspectable.

# %%
if HAS_PANDAS:
    from regression_records import SerializationError
    from regression_records.serialization import (
        SerializationContext,
        default_registry,
        to_json_tree,
    )
    from regression_records.serialization.pandas_serializer import (
        PandasDataFrameSerializer,
    )

    awkward = pd.DataFrame({"mixed": [1, "two", {"three": 3}]})

    ctx = SerializationContext(data_dir=STORE / "scratch", registry=default_registry)
    try:
        to_json_tree(awkward, ctx)
    except SerializationError as e:
        print(f"refused by default: {str(e)[:100]}...")

    pickling = default_registry.copy()
    pickling.register(PandasDataFrameSerializer(allow_pickle=True), replace=True)
    ctx = SerializationContext(data_dir=STORE / "scratch", registry=pickling)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tree = to_json_tree(awkward, ctx)

    print("stored as:", tree["format"])
    for w in caught:
        print(f"{w.category.__name__}: {str(w.message)[:110]}...")

# %% [markdown]
# ## Two things to know about frames on disk
#
# * **Parquet is content-addressed**, so a frame recorded identically by ten different
#   functions is one file per identity directory - and re-recording an unchanged frame
#   writes nothing.
# * **`DatetimeIndex.freq` does not survive parquet.** An index round-trips with its
#   values and dtype but not its inferred frequency. If `freq` is part of what you are
#   testing, assert on it separately.

# %%
print("parquet files:", sorted(p.name for p in STORE.rglob("*.parquet")))
