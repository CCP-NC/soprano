#!/bin/bash

SOPRANO_DIR=../../soprano

find *.rst | grep -v "index.rst" | xargs rm
sphinx-apidoc -e -f -o doctree $SOPRANO_DIR -M -d 5
