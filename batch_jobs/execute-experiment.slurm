#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24000
#SBATCH --job-name="Run experiment"
#SBATCH --output=batch_jobs/output/sbatch_job.txt
#SBATCH --mail-user=sindrejohan1@gmail.com
WORKDIR=${SLURM_SUBMIT_DIR}
OUTPUT_DIR=${SLURM_SUBMIT_DIR}/batch_jobs/output

cd ${WORKDIR}
export EXPERIMENT_TITLE="arima_prediction_dataset1"
export EXPERIMENT_DESCRIPTION="Execute-prediction-on-test-set-on-dataset1-20-correlating-categories"

echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

echo "--Files in current directory--"
ls -la
module purge
module load Python/3.9.6-GCCcore-11.2.0

export NEPTUNE_API_TOKEN=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzYmNjY2MzMC1lMWRmLTQ4M2MtOTdkMy1iZDBlNzg2OGU1NTAifQ==

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python src/main.py --experiment ${EXPERIMENT_TITLE} ${EXPERIMENT_DESCRIPTION} --save

