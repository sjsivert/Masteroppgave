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
echo "variable 1: $1"
echo "variable 2: $2"
echo "variable 3: $3"
echo "Experiment title: ${EXPERIMENT_TITLE}"
ls
module purge
module load Anaconda3/2020.07

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
#pip install papermill


python src/main.py --experiment ${EXPERIMENT_TITLE} ${EXPERIMENT_DESCRIPTION}
