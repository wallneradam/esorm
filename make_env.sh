#!/bin/sh
#
# Make virtualenv for development
#

PYTHON="python3.11"

if ! [ -x "$(command -v virtualenv)" ]; then
    echo "Virtualenv is not installed. Please install virtualenv and try again."
    exit 1
fi

if ! [ -x "$(command -v "$PYTHON")" ]; then
    echo "$PYTHON is not installed. Please install $PYTHON and try again."
    exit 2
fi

if [ ! -d "venv" ]; then
    echo "Creating virtualenv..."
    virtualenv -p "$PYTHON" venv >/dev/null
    if [ $? -ne 0 ]; then
        echo "Failed to create virtualenv!"
        exit 3
    fi
    echo "Virtualenv created."
fi

# Activate virtualenv
. venv/bin/activate

# Install dependencies and package in development mode into virtualenv
echo "Installing dependencies and package..."
python -m pip install --upgrade pip >/dev/null 2>&1
python -m pip install -e ".[dev]"

deactivate

echo ""
echo "Run 'source venv/bin/activate' to enter the virtualenv."
