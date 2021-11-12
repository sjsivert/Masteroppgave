#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:00:02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="Python test"
#SBATCH --output=python-test.log
#SBATCH --mail-user=sindrejohan1@gmail.com

echo "Hello World"
module purge
module load Anaconda3/2020.07

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install papermill

python batch_jobs/test-job.py
