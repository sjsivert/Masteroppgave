#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:00:02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="Run papermill"
#SBATCH --output=batch_jobs/output/sbatch_job.log
#SBATCH --mail-user=sindrejohan1@gmail.com

WORKDIR=${SLURM_SUBMIT_DIR}
OUTPUT_DIR=${SLURM_SUBMIT_DIR}/Masteroppgave/batch_jobs/output

cd ${WORKDIR}

echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

echo "Hello world!"
ls
