#!/bin/bash

SOPRANO_DIR=../../soprano

rm doctree/*.rst
sphinx-apidoc -e -f -o doctree $SOPRANO_DIR -M -d 5
