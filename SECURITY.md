# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately, through GitHub's
[private vulnerability reporting](https://github.com/ronaldBuddys/regression-records/security/advisories/new)
(the **Security** tab, then **Report a vulnerability**). Do not open a public issue.

You should get a reply within a week. Please include a minimal store or script that
reproduces the problem, if you have one.

## Supported versions

Only the latest commit on `main` is supported. The package is not on PyPI and has no
release branches, so fixes land on `main` and nowhere else.

## Trust model

**A record store is code-adjacent data: never load a store you did not write, or did
not review as you would review code.** Loading is not inert.

| What loading does | Why it matters | What guards it |
| --- | --- | --- |
| Imports the module a recorded object names, then allocates that class with `object.__new__` and sets the recorded attributes | Importing a module runs its top-level code, and the record chooses the module | Nothing beyond a check that the name resolves to a class. Review store changes as you review code |
| `replay` imports each recorded function's module, then calls the function on the recorded arguments | Importing runs the module's top-level code, and the record chooses the module and the arguments | Replay calls only a function that is still wrapped by `record_func` / `record_method` under the recorded name; anything else is reported, not called |
| Reads `.npy` side files | A crafted `.npy` could otherwise carry a pickle | `np.load(..., allow_pickle=False)`, always |
| Reads pandas side files | Unpickling runs arbitrary code | Pickle is opt-in on both write and read (`allow_pickle=True` on the serializer); a record's `format` field cannot ask for it by itself. With `verify_side_files=True` the pickle is checked against its content hash before it is unpickled |
| Resolves side-file references | A reference could point outside the store | Absolute paths and `..` are refused with `CorruptRecordError` |

Side files are not hash-checked on load by default, so a modified or swapped side file
loads as if it were the recorded value. `RecordStore(root, verify_side_files=True)`
checks every side file it loads, pickles included, and raises `CorruptRecordError` on a
mismatch.

## What a store can reveal

A record holds every recorded argument and return value in full, and file permissions
follow your umask. Some records also hold details of the machine that wrote them:

- a recorded exception keeps its traceback, which names **absolute local paths**;
- a run manifest (`session(...)`) records the platform string, the install source of
  each named package (a local path, for a local or editable install) and the path of
  each named git repository.

Do not record calls whose arguments carry secrets, and read a store before you publish
it.

See [the Trust section of the persistence docs](docs/persistence.md#trust) for the
details.
