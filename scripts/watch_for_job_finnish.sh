#!/bin/bash

LOG_FILE=batch_jobs/output/sbatch_job.txt
EMAIL=sindrejohan1@gmail.com
JOB_NAME=$1

# Load environment variables from .env file
source ./.env
echo "Job name: $1"
echo "Experiment description $2"
echo "Job id: $2"
JOB_NAME=$1
JOB_ID=$3

function checkJobStatus {
	JOBS="$(squeue | grep $JOB_ID)"
	JOBS_LENGTH=${#JOBS}
	JOB_TIMER="$(echo $JOBS | grep -o -P '(.[0-9]:[0-9][0-9])')"
	if [ $JOBS_LENGTH -ne 0 ] && [ $JOB_TIMER == "0:00" ] ; then
		echo "Waiting for job to start"
	elif [ $JOBS_LENGTH -ne 0 ] && [ $JOB_TIMER != "0:00" ] ; then
		echo "Job started. Timer $JOB_TIMER"

	else
		echo "Job is finished"
	fi

}
checkJobStatus
EMAIL_NOTIFICATION_SENT=0
JOB_STATUS=$(checkJobStatus)
while [[ ! $JOB_STATUS =~ .*finished*. ]];do
	checkJobStatus
	JOB_STATUS=$(checkJobStatus)
	if [[ $JOB_STATUS =~ .*started*. ]] && [ $EMAIL_NOTIFICATION_SENT == 0 ] ; then
		EMAIL_NOTIFICATION_SENT=1
		echo "Job started $JOB_NAME, id: $JOB_ID" | mail -s $JOB_NAME $EMAIL
	fi
	sleep 5
done


cat "Job finnished" | mail -s $JOB_NAME $EMAIL
cat $LOG_FILE | mail -s $JOB_NAME $EMAIL

git add .
#git checkout -b HPC-JOB/$JOB_NAME
git commit -m "E:HPC: $JOB_NAME"
#git push --set-upstream origin HPC-JOB/$JOB_NAME
git push

