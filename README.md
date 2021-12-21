[![Coverage Status](https://coveralls.io/repos/github/NikZy/Masteroppgave/badge.svg?branch=master)](https://coveralls.io/github/NikZy/Masteroppgave?branch=master)[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=NikZy_Masteroppgave&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=NikZy_Masteroppgave)
# Project basics
The basics of how to set up and run this project

## Install project
From project root run:

Setup up python virtual env:
`python -m venv env && source env/bin/activate`

Install project and dependencies
`pip install -r requirements.txt`

## Running experiments
Run `python src/main.py --help` to get a help message.

Run an experiment `python src/main.py -e <experiment-title-wihout-spaces> "<description of the experiment>"`
## Running tests
From project root run:
`mamba`

External API tests takes a long time to run. To just run the fast local tests run:
`mamba -t unit,integration`

## Formatting source files to code formatting standards
Run `black src spec`

# Execute jobs on NTNU High Perfomance Computing cluster
1. Write code for the experiment
2. Add or tune a .slurm file in batch_Jobs/
3. Run ´./scripts/execute_experiment_hpc.sh <job-file.slurm> <Job Name> <Job description>´

# Environment variables
Create a .env file in root folder to configure project related environment variables

```
# Email used to send HPC batch job status emails
EMAIL=<email>
LOG_FILE=<batch_job_log_file> default: batch_jobs/output/sbatch_job.txt
NEPTUNE_API_TOKEN=<api-token>
```

# Folder structure

```
├── .pylintrc          <- Python style guidance
├── README.md          <- The top-level README for developers using this project.
├── .env               <- Environment variables
│
│
├── batch_jobs		   <- Batch .slurm files for executing workloads on HPC-cluster
│   └── output         <- Log output from batch_jobs
│
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   ├── main.py		   <- Main file for project
│   ├── experiment.py  <- Experiment class
│   │
│   │
│   ├── data_types     <- Data classes, enums and data types.
│   │
│   ├──mode-_structures<- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── ...
│   │   └── ...
│   │
│   ├── pipelines      <- Data processing pipelines
│   │
│   ├── utils		   <- Utilities and helper functinos
│   │   ├── config_parser.py
│   │   ├── ...
│   │   └── logger.py
│   │
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── ...
│
├── spec/		       <- Spesification tests for the project
└── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
```
