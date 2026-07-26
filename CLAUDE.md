# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A personal, self-paced walkthrough of an MLOps course (NYC taxi trip duration prediction). Work is organized
as numbered module folders, each corresponding to a course section, containing Jupyter notebooks that are run
top-to-bottom manually. There is no application code, package, CLI, build system, or test suite — everything
is notebook-driven experimentation.

## Layout

- `01-intro/` — environment sanity-check notebook (pandas/sklearn versions).
- `02-ml-model/` — core modeling notebook (`duration-prediction.ipynb`): loads green taxi trip parquet data,
  engineers `duration` and categorical/numerical features, trains Linear Regression/Lasso/XGBoost models, and
  logs runs to MLflow. `deploy-prediction.ipynb` covers the MLflow *model registry* side: searching runs,
  registering a model, transitioning it between stages, downloading a registered model's preprocessor
  artifact, and evaluating it against held-out data.
- `03-experiment-tracking/` — holds the MLflow tracking store (`mlflow.db`, a sqlite file) used by the
  notebooks in `02-ml-model/`, plus its own `requirements.txt`.
- `data/` — raw `green_tripdata_YYYY-MM.parquet` NYC TLC trip files (2021-01, 02, 03), read directly by
  notebooks via relative paths like `../data/green_tripdata_2021-01.parquet`.
- `models/` and `models_pickle/` — pickled `(DictVectorizer, model)` tuples exported from notebooks
  (e.g. `models/lin_reg.bin`).
- `02-ml-model/mlruns/` — MLflow's local artifact store (separate from the sqlite metadata db).

## Working with the notebooks

- Notebooks assume the working directory is their own folder and reference sibling data via `../data/...` —
  run them in place, don't move them.
- MLflow tracking URI is hardcoded as an absolute sqlite path:
  `sqlite:////home/ubuntu/Mlops-course/03-experiment-tracking/mlflow.db`. If the repo is relocated, this path
  breaks and must be updated in every notebook cell that calls `mlflow.set_tracking_uri` /
  `MlflowClient(tracking_uri=...)`.
- The MLflow experiment name in use is `nyc_taxi_experiment`; the registered model name is
  `nyc-taxi-regressor`.
- Feature pipeline convention (repeated across notebooks): parse `lpep_pickup_datetime`/
  `lpep_dropoff_datetime`, compute `duration` in minutes, filter to `1 <= duration <= 60`, cast
  `PULocationID`/`DOLocationID` to string, vectorize `[PULocationID, DOLocationID, trip_distance]` dicts with
  `DictVectorizer`. Keep new feature-engineering notebooks consistent with this if extending the same
  experiment lineage.
- Dependencies for the experiment-tracking module are listed in `03-experiment-tracking/requirements.txt`
  (`mlflow`, `jupyter`, `scikit-learn`, `pandas`, `seaborn`, `hyperopt`, `xgboost`); there is no root-level
  requirements file, so install from that file when setting up an environment for these notebooks.
- To inspect or launch the MLflow UI against the tracking store: `mlflow ui --backend-store-uri sqlite:////home/ubuntu/Mlops-course/03-experiment-tracking/mlflow.db`.

## Data and generated artifacts

`mlflow.db`, `mlruns/`, parquet files under `data/`, and pickled models are runtime/experiment outputs, not
source code — treat changes to them as data, not logic, when reviewing diffs.
