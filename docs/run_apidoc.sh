#!/bin/bash


find *.rst | grep -v "index.rst" | xargs rm
sphinx-apidoc -e -f -o . ../soprano -M -d 5
