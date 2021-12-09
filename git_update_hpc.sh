#!/bin/bash
echo "Job: $1"
echo "Job title: $2"
echo "Job experiment descriptoion $3"

export LOG_LOCATION=batch_jobs/output/sbatch_job.txt
export TERM=xterm


BRANCH_NAME="hpc/$2"

git checkout -b $BRANCH_NAME
git add .
git commit -m "HPC-init: $2"
git push --set-upstream origin $BRANCH_NAME

USERNAME=${username:=sjsivert}

ssh -t ${username:=sjsivert}@login.stud.ntnu.no "ssh -t idun 'cd Masteroppgave && \
		ls && \
		git fetch && \
		git stash && \
		git checkout $BRANCH_NAME && \
		touch $LOG_LOCATION && \
		> $LOG_LOCATION && \
		echo "$2: $3">> $LOG_LOCATION && \
		echo "Adding job to queue" && \
		export EXPERIMENT_TITLE=$2 && \
		export EXPERIMENT_DESCRIPTION=$3 && \
		sbatch $1 && \
		export JOB_ID=$(sbatch $1 | grep -o -P '(\d{7})') && \
		screen -d -m sh watch_for_job_finnish.sh $2 $USERNAME && \
		squeue -u $USERNAME'"
