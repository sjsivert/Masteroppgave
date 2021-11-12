echo "Job: $1"
echo "Files: $2"

ssh -t sjsivert@login.stud.ntnu.no "ssh -t idun 'cd Masteroppgave && \
		git pull && \
		sbatch $1 && \
		squeue -u sjsivert && \
		bash -l'"
