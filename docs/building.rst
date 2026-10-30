Building the docs locally
=========================

The docs are built with Sphinx from a clone of the repository. ``uv`` installs
everything, including the ``docs`` extra (Sphinx, Sphinx-Gallery, MyST, the theme,
``sphinxcontrib-bibtex`` for citations and matplotlib for figures):

.. code-block:: bash

   uv sync --all-extras
   uv run make -C docs html
   open docs/_build/html/index.html

``--all-extras`` rather than ``--extra docs``, because the dataframe example needs polars
and pandas. ``make html`` passes ``-W`` to ``sphinx-build``, so a warning - a broken
cross-reference, malformed reStructuredText in a docstring, a gallery script without a
title - fails the build instead of being buried in the log. CI runs the same command on
every pull request.

Running the examples
--------------------

Examples execute at build time by default: they write to throwaway stores under the
system temp directory and need nothing that is not in the repository.
``DOCS_RUN_EXAMPLES=0`` renders them without running, which is faster and is what a
docs-only change wants.

.. code-block:: bash

   DOCS_RUN_EXAMPLES=1 uv run make -C docs html   # execute every example
   DOCS_RUN_EXAMPLES=0 uv run make -C docs html   # render only

Everything Sphinx-Gallery generates (``docs/auto_examples/``,
``docs/sg_execution_times.rst``) and the HTML under ``docs/_build/`` is ignored by git.
``make -C docs clean`` removes all of it.

Network access
--------------

Intersphinx links numpy and Python names to their own documentation, which means the
build downloads their inventories. Under ``-W`` a failed download fails the build, so
offline, build with ``DOCS_INTERSPHINX=0``: everything still builds, and those names are
simply not linked.

Adding an example
-----------------

Drop a script in the section of ``examples/`` it belongs to, with a two-digit prefix
that places it in the reading order (the numbers run on across sections; a new section
also needs a ``GALLERY_HEADER.rst`` and an entry in ``SUBSECTION_ORDER`` in
``conf.py``). It must start with a module docstring whose first line is an rST title with an
underline of the same length (``tests/test_examples.py`` checks both). The title reads
``<what is used> : <prose>``, with each function, class or module name as an inline
literal so it renders as code, for example::

   ``replay`` : catching a regression
   ==================================

The underline counts the backticks too. The page heading and the table of contents show
the name as code, and the gallery thumbnails and browser tab show the plain text.
``# %%`` comment cells split the script into prose and code.

A page is self-contained: its ``.py`` and notebook downloads must run on their own, and
the tests run each page copied alone into an empty directory. Replay imports recorded
code by module and name, so a page cannot record a function or class it defines itself;
it imports the code under test from a module of ``regression_records._examples``, which
ships with the package and is not public API. Give that module a name no other page
uses, and decorate with a bare ``@record_func()``, so it reads its store from
``REGRESSION_RECORDS_STORE`` at import; the page's first cell sets that variable,
before importing anything (the tests run each page without it, so one that imports
first fails). Print what the page calls with
``print(inspect.getsource(...))``: the whole module when it is short, otherwise each
part beside the section that first uses it. Every page draws at least one figure, which
becomes its gallery thumbnail, and every page but the last ends with a ``**Next:**``
link to the one after it.

Checks *assert* (``np.testing.assert_allclose`` with explicit tolerances, or an exact
comparison) rather than print, so a page that stops matching the code fails the build.
Timings belong on a performance page only. A page that states theory cites it with
``:footcite:`key``` from ``docs/references.bib`` and ends with ``.. footbibliography::``;
add a bibliography entry only after checking every field against the publisher or a DOI
resolver.
