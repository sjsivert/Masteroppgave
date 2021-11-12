#!/bin/bash
echo "Job: $1"
echo "Job experiment message: $2"

export LOG_LOCATION=batch_jobs/output/sbatch_job.txt
export TERM=xterm

git add .
git commit -m "HPC-init: $2"
git push
USERNAME=${username:=sjsivert}

ssh -t ${username:=sjsivert}@login.stud.ntnu.no "ssh -t idun 'cd Masteroppgave && \
		git reset --hard origin/master && git pull origin master  && \
		touch $LOG_LOCATION && \
		> $LOG_LOCATION && \
		echo $2 >> $LOG_LOCATION && \
		echo "Adding job to queue" && \
		sbatch $1 && \
		export JOB_ID=$(sbatch $1 | grep -o -P '(\d{7})') && \
		screen -d -m sh watch_for_job_finnish.sh $2 $USERNAME && \
		squeue -u $USERNAME'"
