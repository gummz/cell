import pandas as pd
from os.path import join
import src.data.constants as c

c.setcwd()

# Split path into lists, don't take last item
# so that we  move up one directory, and lastly
# join items of list into one string with ''.join
raw_data_folder = '/'.join(c.RAW_DATA_DIR.split('/')[:-1])
save_location = c.DATA_DIR
excel = pd.read_excel(join(raw_data_folder, c.EXCEL_FILENAME))

excel['Raw data file name'] = excel['Raw data file name'].str.replace(
    ' ', '_').str.replace('.lsm', '')
excel['Comment'] = excel['Comment'].str.lower()

excel.to_csv(join(c.DATA_DIR, c.EXCEL_FILENAME))

print('extract_raw_info.py complete')
