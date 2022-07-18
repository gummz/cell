import json
import os.path as osp
import src.data.constants as c
import src.data.utils.utils as utils
import os
from time import time

if __name__ == '__main__':
    modes = ('train', 'val', 'test')
    db_version = 'hist_eq'

    tic = time()
    utils.set_cwd(__file__)
    data_dir = c.DATA_DIR if osp.exists(c.DATA_DIR) else c.PROJECT_DATA_DIR
    for mode in modes:
        json_path = osp.join('..', data_dir,
                             'db_versions', db_version, mode,
                             c.IMG_DIR)
        json_files = (file for file in os.listdir(json_path)
                      if file.endswith('.json'))

        for json_file in json_files:
            load_path = osp.join(json_path, json_file)
            with open(load_path, 'r') as jsonFile:
                data = json.load(jsonFile)

            data['imagePath'] = data['imagePath'].replace('_', '')

            with open(load_path, 'w') as jsonFile:
                json.dump(data, jsonFile)

    print(f'rename_json.py complete in {utils.time_report(time, tic())}.')
