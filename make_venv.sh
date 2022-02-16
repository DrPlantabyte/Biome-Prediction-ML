#!/bin/bash
## Example:
## $ ./make_venv.sh env requirements.txt

# exit when any command fails
set -e

# wrong number of arguments
if [ "$#" -ne 2 ]; then
    echo "Wrong arguments. Usage:"
    echo "$0 venv_name requirements_file"
    exit 1
fi

VENV_NAME="$1"
REQ_FILE="$2"

if test ! -d "$VENV_NAME"; then
	python3 -m venv "$VENV_NAME"
	echo "*" > "$VENV_NAME"/.gitignore
	source "$VENV_NAME"/bin/activate
	pip install --upgrade pip
	pip install -r "$REQ_FILE"
	pip freeze > "VENV_NAME"/requirements.txt
	deactivate
else
	echo "Directory '$VENV_NAME' already exists. No action taken. Either delete this directory or use '$VENV_NAME/bin/pip install -r $REQ_FILE' instead."
	exit 1
fi

