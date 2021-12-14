#!/bin/bash
echo "Experiment title: $1"
echo "Experiment message: $2"

export TERM=xterm

# Run experiment
python ./src/main.py -e $1 $2

# Add experiment commit
git add .
git commit -m "E: $1"
