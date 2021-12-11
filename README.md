# Execute jobs on NTNU High Perfomance Computing cluster
1. Write code for the experiment
2. Add or tune a .slurm file in batch_Jobs/
3. Run ´./git_update_hpc.sh <job-file.slurm> <Job Name>´

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
├── LICENSE
├── .pylintrc          <- Python style guidance
├── Makefile           <- Makefile with commands like `make data` or `make train`
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
