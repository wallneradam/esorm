#!/bin/sh

# Tests in these python interpreters
#PYTHONS="python3.8 python3.9 python3.10 python3.11"
PYTHONS="python3.11"

# Check if docker is installed
if ! [ -x "$(command -v docker)" ]; then
    echo "Docker is not installed. Please install docker and try again."
    exit 1
fi

# Check if virtualenv is installed
if ! [ -x "$(command -v virtualenv)" ]; then
    echo "Virtualenv is not installed. Please install virtualenv and try again."
    exit 2
fi

# Check if pythons are installed
for PYTHON in $PYTHONS; do
    if ! [ -x "$(command -v "$PYTHON")" ]; then
        echo "$PYTHON is not installed. Please install $PYTHON and try again."
        exit 3
    fi
done

# Run test in each python interpreter
for PYTHON in $PYTHONS; do
    echo "*** Testing in $PYTHON ***"

    # Create virtualenv
    virtualenv -p "$PYTHON" venv_test >/dev/null
    . venv_test/bin/activate

    echo " - Installing dependencies..."
    pip install --upgrade pip >/dev/null 2>&1
    pip install -e ".[dev]" >/dev/null
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies in $PYTHON!"
        exit 4
    fi

#    if [ $? -ne 0 ]; then
#        echo "Tests failed in $PYTHON!"
#        exit 1
#    fi

    # Remove virtualenv
    deactivate
    rm -rf venv_test
done
