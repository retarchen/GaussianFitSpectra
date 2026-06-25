# Minimal Sphinx Makefile for gaussFitSpec documentation.

SPHINXOPTS    ?=
SPHINXBUILD   ?= LC_ALL=C LANG=C MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/fontcache /Users/retarchen/miniconda3/envs/codeastro/bin/python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
