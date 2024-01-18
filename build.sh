#!/bin/sh

rm dist/*.tar.gz

# Build the project
python setup.py sdist
if [ $? -ne 0 ]; then
    echo "Build failed!" >&2
    exit 1
fi

# Build documentation
( cd docs && make clean && make html )
if [ $? -ne 0 ]; then
    echo "Doc build failed!" >&2
    exit 2
fi

if [ "$1" = "upload" ]; then
    # Upload to PyPI
    twine upload dist/*
    if [ $? -ne 0 ]; then
        echo "Upload failed!" >&2
        exit 3
    fi
fi