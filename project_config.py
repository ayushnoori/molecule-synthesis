'''
PROJECT CONFIGURATION FILE
This file contains the configuration variables for the project. The variables are used 
in the other scripts to define the paths to the data and results directories. The variables 
are also used to set the random seed for reproducibility.
'''

# import libraries
import os
from pathlib import Path

# check if on O2 or not
home_variable = os.getenv('HOME')
on_remote = (home_variable == "/home/an252")

# define base project directory based on whether on O2 or not
if on_remote:
    PROJECT_DIR = Path('')
else:
    PROJECT_DIR = Path('/Users/an583/Library/CloudStorage/OneDrive-Personal/Academic/College/Junior Year/Fall Term/COMPSCI 252R/molecule-synthesis')

# define project configuration variables
DATA_DIR = PROJECT_DIR / 'Data'
RESULTS_DIR = PROJECT_DIR / 'Results'
SEED = 42