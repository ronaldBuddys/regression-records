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
# # 1. Recording your first call
#
# `record_func` needs two things: a function, and somewhere to keep its history.
#
# The first time each distinct set of arguments is seen, the call is written to disk as a
# **regression record**. Every later run of that same call is compared against the
# recording. Nothing else about the function changes - arguments arrive untouched, the
# return value passes through.

# %%
import sys
from pathlib import Path

# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd()))

import numpy as np

from _shared.demo import fresh_store, show_record, show_tree
from regression_records import record_func

STORE = fresh_store("01_first_record")
STORE

# %% [markdown]
# ## Decorate and call
#
# `store` is mandatory. There is no default location, because recording somewhere
# implicit would be worse than being asked.

# %%
@record_func(STORE)
def summarize(values, *, scale=1.0):
    values = np.asarray(values) * scale
    return {"n": int(values.size), "mean": float(values.mean()), "values": values}


result = summarize([1.0, 2.0, 3.0, 4.0])
result

# %% [markdown]
# ## What landed on disk
#
# One JSON record, plus a side file for the array. Arrays and dataframes are too big and
# too binary to sit inside JSON, so they are written next to it and referenced by a
# relative path - which is what lets the whole directory be moved or copied as a unit.

# %%
show_tree(STORE)

# %% [markdown]
# The directory path is the function's identity: its module, then its qualified name.
# Running this file as a script makes the module `__main__`; imported from a real module
# it would be `mypkg/analysis/summarize/` instead.

# %%
show_record(STORE, keys=["identity", "input_hash", "invocation", "output"])

# %% [markdown]
# Worth noticing in that JSON:
#
# * **`invocation.args` and `invocation.kwargs`** hold the arguments *after* binding them
#   to the signature. `scale` was never passed, but it is recorded as `1.0` - so
#   `summarize([1.0, 2.0])`, `summarize([1.0, 2.0], scale=1.0)` and a future call that
#   relies on the same default are all recognised as the same case.
# * **`input_hash`** is a sha256 over that normalized argument tree, and it is the lookup
#   key. It is also the leading part of the record's filename, so finding the record for a
#   call is a directory glob rather than a scan of every record.
# * **`identity.code_hash`** is recorded but is *not* part of the key. Editing the
#   function does not orphan its history; it produces a warning instead (example 7).

# %% [markdown]
# ## The second call validates
#
# Same arguments, so this call is compared against the record rather than recorded again.
# A passing comparison is silent - the function just returns.

# %%
summarize([1.0, 2.0, 3.0, 4.0])
print("passed:", summarize.last_result.passed)
print("records on disk:", len(list(STORE.rglob("rec-*.json"))))

# %% [markdown]
# ## Different arguments are a different case
#
# A new set of arguments has no history to compare against, so it becomes a second
# recorded case. A function accumulates one record per distinct invocation.

# %%
summarize([1.0, 2.0, 3.0, 4.0], scale=2.0)
summarize([10.0, 20.0])
show_tree(STORE)

# %% [markdown]
# `summarize.last_result` is `None` after a call that *recorded* - there was nothing to
# compare against. It holds a `ComparisonResult` only after a call that validated.

# %%
print(summarize.last_result)
