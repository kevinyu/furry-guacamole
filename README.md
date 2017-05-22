# furry-guacamole

## Setup

### Environment

* Create a new conda environment, `conda create --name bird python`

* Setup paths using handy script: `. setup.sh`

* Or, if not using conda, just makes sure `PROJECT_ROOT` and `PYTHONPATH` env vars get setup appropriately

### Dependencies

* While in the environment, install depenencies from requirements.txt

### Config

* Copy `src/config_default.py` to `src/config.py` and fill in `DATADIR` to be the path to where your h5 data is (this directory should have directories with bird names at the top level)
