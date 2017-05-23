import os


PROJECT_ROOT = os.path.join(os.getenv("PROJECT_ROOT"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# SCRIPT PARAMS
MIN_TIME = 0.0  # seconds
MAX_TIME = 0.6
FILTER_TYPE = "exp"
FILTER_WIDTH = 5.0
DIMS = [2, 5, 13, 34]
# DIMS = [1, 2, 3, 5, 8, 13, 21, 34, 55]
