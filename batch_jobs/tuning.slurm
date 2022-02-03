#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="Run Tuning"
#SBATCH --output=batch_jobs/output/sbatch_job.log
#SBATCH --mail-user=sander.k.kilen@gmail.com

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

echo ""
echo "variable 1: $1"
#echo "variable 2: $2"
#echo "variable 3: $3"
echo "Experiment title: ${EXPERIMENT_TITLE}"
ls
module purge
module load Python/3.8.6-GCCcore-10.2.0

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python src/main.py --experiment ${EXPERIMENT_TITLE} ${EXPERIMENT_DESCRIPTION}
