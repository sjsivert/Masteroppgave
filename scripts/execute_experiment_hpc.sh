#!/bin/bash
function getApiCount {
    count=$(curl https://api.countapi.xyz/hit/default/65898bf3-085e-4339-b66f-bef550fc9218 | grep -o -P '\d+')
}
getApiCount
experiment_title="$2$count"
echo "Job: $1"
echo "Job title: $experiment_title"
echo "Job experiment description $3"


export LOG_LOCATION=batch_jobs/output/sbatch_job.txt
export TERM=xterm


BRANCH_NAME="hpc/$experiment_title"

echo "Checkout branch $BRANCH_NAME"
git checkout -b $BRANCH_NAME
git add .
git commit -m "E:HPC-init $experiment_title: $3"

git push --set-upstream origin $BRANCH_NAME

USERNAME=${username:=sjsivert}

ssh -t ${username:=sjsivert}@login.stud.ntnu.no "ssh -t idun 'cd Masteroppgave && \
		ls && \
		git fetch && \
		git stash && \
		git checkout $BRANCH_NAME && \
		touch $LOG_LOCATION && \
		> $LOG_LOCATION && \
		echo "$experiment_title: $3">> $LOG_LOCATION && \
		echo "Adding job to queue" && \
		export EXPERIMENT_TITLE=$experiment_title && \
		export EXPERIMENT_DESCRIPTION=$3 && \
		sbatch $1 && \
		export JOB_ID=$(sbatch $1 | grep -o -P '(\d{7})') && \
		screen -d -m sh ./scripts/watch_for_job_finnish.sh $experiment_title $3 $JOB_ID && \
		squeue -u $USERNAME'"
