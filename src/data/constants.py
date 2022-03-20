# CONNECTED COMPONENT ANALYSIS
CV2_CONNECTED_ALGORITHM = 1
NUMBER_CONNECTIVITY = 8

# DATA CONSTANTS
RAW_DATA_DIR = '/dtu-compute/tubes/raw_data'
# Very important to maintain the order of files in `RAW_FILES`
# because the dataset creation depends on this order
# Already processed: First two
RAW_FILES = ['LI_2019-01-17_emb7_pos3.lsm',
             'LI_2020-05-06_emb7_pos3.lsm',
             'LI_2019-07-03_emb7_pos3.lsm',
             'LI_2018-12-07_emb6_pos3.lsm',
             'LI_2018-12-18_emb4_pos4.lsm',
             'LI_2020-05-06_emb7_pos4.lsm',
             'LI_2018-12-07_emb5_pos2.lsm']
generalize = ['LI_2019-11-21_emb6_pos3.lsm',
              'LI_2019-02-05_emb5_pos4.lsm']
items = generalize[1]
if type(items) == str:
    RAW_FILES_GENERALIZE = [generalize[1]]
elif type(items) == list:
    RAW_FILES_GENERALIZE = generalize[1]

# RAW_FILES = RAW_FILES_GENERALIZE  # testing for generalization
#  'LI_2020-06-17_emb3_pos5.czi']
START_IDX = 0


# for consideration:

# Time points at which the file should be cut off
# because the data isn't usable past that point
cutoffs = [None, None, 250, None, None, None, None]
RAW_CUTOFFS = dict(zip(RAW_FILES, cutoffs))
# 'LI_2020-06-17_emb7_pos3.lsm'
# ^ changed file extension to .cri

# z-dimensions of the raw data files, in order
idx = [38, 33, 29, 33, 36, 33, 33]
RAW_FILE_DIMENSIONS = dict(zip(RAW_FILES, idx))

# number of timepoints of the raw data files, in order
TIMEPOINTS = [280, 288, 295, 276, 290, 288, 276]

# To be used within the src/[subfolder] directories
DATA_DIR = '../../data/interim/'
IMG_DIR = 'imgs'
MASK_DIR = 'masks'
FIG_DIR = 'figures'

# IMAGE FILTERING
BLOCK_SIZE = 5
C = 14
MEDIAN_FILTER_KERNEL = 9
SIMPLE_THRESHOLD = 80


# Index for first two raw data files was: 579
# How often to print out with matplotlib
# for make_dataset.py and annotate.py
DBG_EVERY = 70
