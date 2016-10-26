#!/bin/bash


find *.rst | grep -v "index.rst" | xargs rm
sphinx-apidoc -e -f -o doctree ../soprano -M -d 5
