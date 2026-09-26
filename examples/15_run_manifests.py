"""
Run manifests : provenance for each recording session
=====================================================

A record says what a call did, but not what it was made with: which commit of the code,
which package versions, which version of the input data. And because a record's id is
derived from its behavior, a run that re-records identical behavior writes nothing -
the store alone cannot tell you it ever happened.

A ``session`` fills that gap. Every record it sees - written by it (``new``) or already
on disk (``matched``) - is listed in a manifest, with the environment and whatever
context you pass, written to ``<store>/_runs/`` when the block ends. Records are not
touched, and replay never reads manifests.
"""

import inspect
import json
import sys
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

from _shared import runs
from _shared.demo import fresh_store, run_in_fresh_process, show_tree

from regression_records import environment, replay
from regression_records.manifest import iter_manifest_paths, load_manifest

STORE = fresh_store(runs.STORE.root.name)

# %%
# A recorded pipeline
# -------------------
#
# ``session`` does not switch recording on, so it is paired with ``recording()`` (or
# ``REGRESSION_RECORDS=1``). ``context`` is for what the environment cannot know - here
# the version of the data read, and a seed. ``packages`` names distributions to
# describe (version, and the commit for a git install); ``repos`` would name
# repositories, whose ``HEAD`` and dirty flag are recorded.

print(inspect.getsource(runs))

# %%
# A first run
# -----------
#
# Both records are ``new``, and the manifest sits in ``_runs/``, outside every
# ``records/`` directory.

first = runs.run_pipeline([100.0, 101.5, 99.0, 102.0], data_version=34)
show_tree(STORE)

manifest = load_manifest(first)
print(json.dumps({k: manifest[k] for k in ("context", "counts", "records")}, indent=2))
print(json.dumps(manifest["environment"]["packages"], indent=2))

# %%
# The same run again
# ------------------
#
# In a fresh process the same calls behave the same, so no record is written. The
# second manifest still says the run happened, and that it confirmed both records.

AGAIN = """
from _shared import runs
from regression_records.manifest import load_manifest
path = runs.run_pipeline([100.0, 101.5, 99.0, 102.0], data_version=34)
print(load_manifest(path)["counts"])
"""
run_in_fresh_process(AGAIN)

# %%
# New data
# --------
#
# A run on other inputs adds records, and its manifest says which, and on what data.
# A later replay failure on one of them can be traced to the run, and the data
# version, that produced it.

runs.run_pipeline([100.0, 100.5, 101.0], data_version=35)

for path in iter_manifest_paths(STORE):
    m = load_manifest(path)
    print(path.name, m["context"], m["counts"])

# %%
# Saving a replay
# ---------------
#
# The other half of the history: ``ReplayReport.save`` writes a replay's results beside
# the environment it ran in, so a series of saved reports says under which code each
# record stopped holding.

report = replay(runs.STORE)
saved = report.save(STORE / "_replays" / "latest.json", environment=environment())
print(report.summary())
print(sorted(json.loads(saved.read_text())))

# %%
# Manifests and saved reports sit outside every ``records/`` directory, so neither
# replay nor ``prune`` sees them. Commit them with the store to keep the history, or
# delete them freely: no record depends on them.
