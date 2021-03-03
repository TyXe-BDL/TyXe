# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
APIDOC        = sphinx-apidoc
SPHINXPROJ    = TyXe
SOURCEDIR     = source
TEMPLATEDIR   = "$(SOURCEDIR)/_templates"
PROJECTDIR    = ../tyxe
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

apidoc:
	$(APIDOC) --templatedir "$(TEMPLATEDIR)" -fMeT -o "$(SOURCEDIR)" "$(PROJECTDIR)"

# hack to avoid trailing module/package headers from: 
# https://stackoverflow.com/questions/21003122/sphinx-apidoc-section-titles-for-python-module-package-names
docs:
	$(MAKE) clean
	$(MAKE) apidoc
	$(MAKE) html

clean:
	@echo "Removing everything under 'build' and 'source/generated'.."
	@rm -rf $(BUILDDIR)/html/ $(BUILDDIR)/doctrees $(SOURCEDIR)/generated $(SOURCEDIR)/_autosummary

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
