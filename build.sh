#!/bin/sh

# Build the project
python setup.py sdist
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Build documentation
cd docs
make html
if [ $? -ne 0 ]; then
    echo "Doc build failed!"
    exit 2
fi

# Upload to PyPI
#twine upload dist/*
