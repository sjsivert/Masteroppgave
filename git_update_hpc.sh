echo "Job: $1"
echo "Username: $2"

ssh -t ${username:=sjsivert}@login.stud.ntnu.no "ssh -t idun 'cd Masteroppgave && \
		git pull && \
		sbatch $1 && \
		squeue -u sjsivert && \
		tail -f batch_jobs/output/sbatch_job.log && \
		bash -l'"
