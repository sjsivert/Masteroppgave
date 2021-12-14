echo "Job: $1"
echo "Files: $2"

echo "Copy job to NTNU network"
rsync --relative $1 sjsivert@login.stud.ntnu.no:~/temp/$1
rsync --relative $2 sjsivert@login.stud.ntnu.no:~/temp/$2

echo "Remote execute add queue job"
ssh -t sjsivert@login.stud.ntnu.no "\
	echo 'Copy files to HPC server' && \
	cd ~/temp/ && \
	rsync $1 sjsivert@idun-login1.hpc.ntnu.no:~/Masteroppgave/$1 && \
	rsync $2 sjsivert@idun-login1.hpc.ntnu.no:~/Masteroppgave/$2 && \
	echo 'Add job to queue' && \
	ssh -t idun 'cd Masteroppgave && \
		sbatch $1 && \
		ls && \
		squeue -u sjsivert'"
