## PROJECT STRUCTURE
_________
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   ├── raw
│   └── results
├── docs
├── figures
│   └── 3d_plot
├── models
├── notebooks
├── references
├── reports
│   └── figures
├── shell
├── src
│   ├── data
│   ├── experiments
│   ├── features
│   ├── models
│   ├── tests
│   ├── tracking
│   └── visualization
└── src.egg-info



`src/models`: where the models are saved
`shell/`: where the shell scripts to run Python scripts on the HPC are saved
`src/experiments`: all automated experiments in the project, like calibration, training grid search, etc.
`src/data`: scripts for creating training, validation, and test sets
`src/tracking`: tracking scripts
`src/visualization`: visualization scripts, like visualizing raw data, 3D prediction points for cells, etc.
`data/`: All the data used in the project. NB. there is also a scratch directory
`notebooks/`: a bunch of scratch notebooks used for development. Probably not useful
`data/interim/db_versions`: Database versions. Currently, vanilla (not processed with histogram equalization) and histogram equalization versions. Histogram equalization version is unfinished.
`data/interim/extract`: Directory for keeping data extracted from the raw data files.

## Script descriptions

`src/data/make_dataset_train/val/test.py`: create dataset used for model training.
`src/data/annotate_from_json.py`: 
