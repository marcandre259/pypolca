# pypolca documentation Makefile

SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= uv run sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help clean doxygen html

help:
	@echo "make html     — Build HTML docs"
	@echo "make clean    — Remove build artifacts"

doxygen:
	doxygen Doxyfile

html: doxygen
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

clean:
	rm -rf build _doxygen
