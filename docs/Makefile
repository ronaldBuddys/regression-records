# Minimal Sphinx makefile. `make html` builds with -W, so a warning is a failure.
# DOCS_RUN_EXAMPLES=1 executes the gallery scripts as well as rendering them.
SPHINXBUILD ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: help html clean
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	rm -rf "$(BUILDDIR)" auto_examples sg_execution_times.rst

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W
