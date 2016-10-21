#!/bin/bash


find *.rst | grep -v "index.rst" | xargs rm
sphinx-apidoc -e -f -o . ../ -M -d 5
