#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:00:02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="Run papermill"
#SBATCH --output=papermill.log
#SBATCH --mail-user=sindrejohan1@gmail.com
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

ls
module purge
module load Anaconda3/2020.07

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install papermill

cd notebooks
papermill sindre_anomely_detection.ipynb ../output/sindre_anomely_detection.ipynb

