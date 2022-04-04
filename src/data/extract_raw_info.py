import pandas as pd
from os.path import join, splitext
import src.data.constants as c
c.setcwd(__file__)

# Split path into lists, don't take last item
# so that we  move up one directory, and lastly
# join items of list into one string with ''.join
raw_data_folder = '/'.join(c.RAW_DATA_DIR.split('/')[:-1])
data_folder = '/'.join(c.DATA_DIR.split('/')[:-2])
print(data_folder)
save_location = c.DATA_DIR
excel = pd.read_excel(join(raw_data_folder, c.EXCEL_FILENAME))

excel['Raw data file name'] = excel['Raw data file name'].str.replace(
    ' ', '_').str.replace('.lsm', '')
excel['Comment'] = excel['Comment'].str.lower()
excel_noext = splitext(c.EXCEL_FILENAME)[0]
excel.to_csv(join(data_folder, f'{excel_noext}.csv'))

print('extract_raw_info.py complete')
