#!/bin/bash
echo "Experiment to run: $1"
echo "Experiment message: $2"

export TERM=xterm

# Run experiment
python $1

# Add experiment commit
git add .
git commit -m "Experiment: $2"
