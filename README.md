[![Coverage Status](https://coveralls.io/repos/github/NikZy/Masteroppgave/badge.svg?branch=master)](https://coveralls.io/github/NikZy/Masteroppgave?branch=master)[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=NikZy_Masteroppgave&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=NikZy_Masteroppgave)

:clipboard: [Issue board](https://linear.app/masterproject/team/MAS/board).
⚗️ [Neptune.ai](https://app.neptune.ai/o/sjsivertandsanderkk/org/Masteroppgave/e/MAS-28/all?path=&attribute=data_pipeline_steps)

# Project basics
The basics of how to set up and run this project

## Install project
Make sure python 3.8 is installed.

From project root run:

Setup up python virtual env:
`virtualenv --python=/usr/bin/python3.x env && source env/bin/activate`


Install project and dependencies
`pip install -r requirements.txt`

## Installing git hooks

`./scripts/install-git-hooks.sh`

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



---------------
# Git conventions

## Commit conventions
When committing new content to git correct and structured conventions should be utilized. In addition to marking commits with the issue number fro 'linear', a commit should always contain a type descriptions (such as for a feature, experiment, test, and so forth),

This inclues the use of prefixes to the commit messages used.
The following are the relevant conventions for this project:

|Prefix |Content |
|------ |------- |
|feat/  | Describes that there is a new feature added |
|test/  | Describes that the contents of the commit is to add tests |
|exp/   | Identification to use in case of an experiment. |
|ref/   | Refactoring code / Rewriting code |
|paper/ | Updates done to the paper |

The commit should start with the type prefix, followed by the MAS code if there is any, lastly followed by a description of the commit function.

|Examples |
|---------|
|feat/MAS-30/Add-functionality-for-saving-figures
|test/Add-missing-tests-with-saving-figures


## Branch naming convention
Branches should follow the same naming convention as the one used for commits.
The branch name is mostly for clarification, and not as important as the commit messages creating the log to be read at a later time.


---------------
# Execute jobs on NTNU High Perfomance Computing cluster

*Prerequisites:*
1. Setup ssh alias in your SSH config for local NTNU user: `<username>@login.stud.ntnu.no`
```
Host idun
	HostName idun-login1.hpc.ntnu.no
	User <username>
```
2. Clone this repo into a `<username>@login.stud.ntnu.no:~/Masteroppgave`
3. Set up SSH keys for IDUN to github to allow pushing back to the repo
4. Configure git config on idun to allow commits
5. Create a .env file in `<username>@login.stud.ntnu.no:~/Masteroppgave/.env`
with the contents described below

--------------

*Running experiments:*

1. Write code for the experiment
2. Add or tune a .slurm file in batch_Jobs/
3. Run `./scripts/execute_experiment_hpc.sh <job-file.slurm> <Job Name> <Job description>`

##Basic batch job commands

* Start job: `sbatch <slurm_file>`
* See jobs: `squeue -u <username>`
* Cancel jobs: `scancel <job-id>`


## SLURM files - Config
In order to execute code on the HPC cluster, the cluster expects the use of a config file in the form of a ".slurm" file.
Such ".slurm" files have already been created and are located under the folder "batch_jobs/".

Before a test is run on the HPC cluster, one should update the current values of the *.slurm files.
The values to be updated are information such as job_name, mail for receiving information, etc.

# Environment variables
Create a .env file in root folder to configure project related environment variables

```
# Email used to send HPC batch job status emails
EMAIL=<email>
USERNAME=<NTNU-username>
LOG_FILE=<batch_job_log_file> default: batch_jobs/output/sbatch_job.txt
NEPTUNE_API_TOKEN=<api-token>
```

# UML Diagram
![UML class](https://user-images.githubusercontent.com/5860069/152807641-6df318db-816a-4bbb-a9fe-a54a5624f90a.png)

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
