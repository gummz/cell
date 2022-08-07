import os.path as osp
import os
import pandas as pd
from datetime import datetime
import src.data.utils.utils as utils
import src.data.constants as c

if __name__ == '__main__':
    utils.set_cwd(__file__)
    raw_files = os.listdir(c.RAW_DATA_DIR)
    df = pd.DataFrame({
        'file': raw_files,
        'last_fetched': (datetime.now() for _ in raw_files)
    })
    save_dir = osp.join('..', c.PROJECT_DATA_DIR, 'raw')
    utils.make_dir(save_dir)
    df.to_csv(osp.join(save_dir, 'raw_files.csv'), index=False)
    print('Saved raw_files.csv')
