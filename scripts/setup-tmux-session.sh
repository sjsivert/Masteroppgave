# !/bin/bash

session="Masteroppgave"

tmux new-session -d -s $session
tmux send-keys -t 'Masteroppgave' 'source .env' C-m 'clear' C-m
tmux send-keys -t  'Masteroppgave' 'source venv/bin/activate' C-m 'clear' C-m
tmux attach-session -t $session

