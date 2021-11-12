echo "Job: $1"
echo "Username: $2"

export LOG_LOCATION=batch_jobs/output/sbatch_job.log

ssh -t ${username:=sjsivert}@login.stud.ntnu.no "ssh -t idun 'cd Masteroppgave && \
		git reset --hard origin && git pull origin master  && \
		sbatch $1 && \
		squeue -u sjsivert && \
		tail -f ${LOG_LOCATION}&& \
		bash -l'"
