import os.path as osp
import cv2


def canny_filter(img, threshold1, threshold2, aperture_size, L2_gradient):
    img = cv2.fastNlMeansDenoising(img, None, 11, 7, 21)
    img_canny = cv2.Canny(
        img, threshold1, threshold2,
        apertureSize=aperture_size, L2gradient=L2_gradient)

    return img_canny


# ACTIVE SLICES RATIO
ACTIVE_SLICES_RATIO = 0.025

ACCEPT_RANGE = (0.95, 1)
MATCH_THRESHOLD = 0.3

# CELL ACTIVATION THRESHOLD
ACTIVE_THRESHOLD = 1e-2

# NEXT CENTER SEARCH RADIUS (pixels)
# (stitching cells together between slices)
SEARCHRANGE = 50

# TRACKING RADIUS (pixels)
# (tracking cells in 3D space)
TRACK_RADIUS = 100

# LOGGING IMAGE SIZE
LOGGING_IMG_SIZE = 512

IMG_SIZE = 1024

# CONNECTED COMPONENT ANALYSIS
CV2_CONNECTED_ALGORITHM = 1
NUMBER_CONNECTIVITY = 8

PRED_FILE = 'LI_2019-01-17_emb7_pos3'

# DATA CONSTANTS
RAW_DATA_DIR = '/dtu-compute/tubes/raw_data'
# Very important to maintain the order of files in `RAW_FILES`
# because the dataset creation depends on this order
RAW_FILES_TRAIN = {
    'LI_2019-11-21_emb6_pos2': None,
    'LI_2019-11-21_emb6_pos3': None,
    'LI_2020-05-06_emb7_pos3': None,
    'LI_2019-07-03_emb7_pos3': (0, 250),
    'LI_2018-12-07_emb6_pos3': None,
    'LI_2018-12-18_emb4_pos4': None,
    'LI_2020-05-06_emb7_pos4': None,
    'LI_2018-12-07_emb5_pos2': None
}

RAW_FILES_VAL = {
    'LI_2020-05-06_emb4_pos2': None,
    'LI_2019-08-30_emb2_pos1': (0, 250),
    # 'LI_2019-02-05_emb5_pos3': None
    # LI_2020-06-17_emb4_pos2.czi
}

RAW_FILES_TEST = {  # None = ok to use any timepoint
    'LI_2015-07-12_emb5_pos1': None,
    'LI_2016-03-04_emb5_pos2': (0, 117),
    'LI_2018-11-20_emb7_pos3': None,
    'LI_2018-11-20_emb6_pos2': None,
    'LI_2019-01-17_emb7_pos3': None,
    'LI_2019-02-05_emb5_pos4': (0, 123),
    'LI_2019-09-19_emb1_pos3': None,
    'LI_2019-11-08_emb3_pos2': None,
    'LI_2019-11-08_emb5_pos4': None,
    'LI_2020-06-04_emb1_pos1': None,
}

RAW_FILES = {
    'train': RAW_FILES_TRAIN,
    'val': RAW_FILES_VAL,
    'test': RAW_FILES_TEST
}

# RAW_FILES = RAW_FILES_GENERALIZE  # testing for generalization
#  'LI_2020-06-17_emb3_pos5.czi']
START_IDX = 0

# Time points at which the file should be cut off
# because the data isn't usable past that point
cutoffs = [None, None, 250, None, None, None, None]
RAW_CUTOFFS = dict(zip(RAW_FILES, cutoffs))
# 'LI_2020-06-17_emb7_pos3.lsm'
# ^ changed file extension to .cri

cutoffs_test = [None, None]
RAW_CUTOFFS_TEST = dict(zip(RAW_FILES_TEST, cutoffs_test))

# z-dimensions of the raw data files, in order
idx = [29, 33, 29, 33, 36, 33, 33]
RAW_FILE_DIMENSIONS = dict(zip(RAW_FILES, idx))

idx_test = [38, 34]
RAW_FILE_DIMENSIONS_TEST = dict(zip(RAW_FILES_TEST, idx_test))

# number of timepoints of the raw data files, in order
TIMEPOINTS = [328, 288, 295, 276, 290, 288, 276]
TIMEPOINTS_TEST = [280, 289]

# absolute project path
PROJECT_DIR = '/zhome/e2/e/154260/cell/'

# To be used within the src/[subfolder] directories
DATA_DIR = '/work3/s203004/data/interim/'
PROJECT_DATA_DIR = '../../data/interim/'
EXPERIMENT_DIR = '../experiments/'

IMG_DIR = 'imgs'
IMG_DIR_TEST = 'imgs'

MASK_DIR = 'masks'
MASK_DIR_FULL = 'masks_full'
MASK_DIR_TEST = 'masks'
MASK_DIR_TEST_FULL = 'masks_full'

PRED_DIR = 'pred'
PRED_DIR_TEST = 'pred'

EXTRACT_DIR = 'extract'

FIG_DIR = 'figures'
FILE_DIR = 'files'

NB_DIR = 'home/gummz/dtu/cell/cell/notebooks'

# IMAGE FILTERING
BLOCK_SIZE = 5
C = 14
MEDIAN_FILTER_KERNEL = 9
SIMPLE_THRESHOLD = 80


# Index for first two raw data files was: 579
# How often to print out with matplotlib
# for make_dataset.py and annotate.py
DBG_EVERY = 70
IMG_EXT = 'png'

CELL_CHANNEL = 0
TUBE_CHANNEL = 1

MODEL_STR = '25_05_16H_51M_45S'
# 29_04_21H_43M_43S
# :::::
#
# '28_05_12H_38M_47S'
EXCEL_FILENAME = 'Muc1-mcherry_MIP-GFP_database_3.xlsx'
SAMPLE_PATH = osp.join(PROJECT_DATA_DIR, 'sample.npy')


FILTERS = {
    'none': None,
    'mean': (cv2.blur, [(5, 5)]),
    'gaussian': (cv2.GaussianBlur, [(5, 5), 0]),
    'median': (cv2.medianBlur, [5]),
    'bilateral': (cv2.bilateralFilter, [9, 50, 50]),
    'nlmeans': (cv2.fastNlMeansDenoising, [None, 11, 7, 21]),
    'canny': (canny_filter, [20, 20, 3, False])
}

# EXCEL_SHEET = pd.read_csv(join(DATA_DIR, EXCEL_FILENAME))

# testset_ext = utils.add_ext(list(test_set.keys()))
# test_set = {ext: value for ext, value
#             in zip(testset_ext, list(test_set.values()))}

# test_set = {
#     'LI_2015-07-12_emb5_pos1': (16),
#     'LI-2018-11-20_emb6_pos1': (104),
#     'LI_2018-11-20_emb7_pos4': (10, 71),
#     'LI_2019-01-17_emb7_pos4': (158, 217),
#     'LI_2019-02-05_emb5_pos2': (191),
#     'LI_2019-02-05_emb5_pos3': (210),
#     'LI_2019-02-05_emb5_pos4': (133),
#     'LI_2019-04-11_emb5_pos1': (157),
#     'LI_2019-04-11_emb8_pos2': (68),
#     'LI_2019-04-11_emb8_pos3': (245),
#     'LI_2019-04-11_emb8_pos4': (229),
#     # 'LI_2019-06-13_emb2_pos1': (204),
#     # 'LI_2019-06-13_emb2_pos2': (32),
#     'LI_2019-07-03_emb1_pos1': (246),
#     'LI_2019-07-03_emb7_pos2': (98),
#     'LI_2019-07-03_emb7_pos3': (99),
#     'LI_2019-07-03_emb7_pos4': (183),
# }
