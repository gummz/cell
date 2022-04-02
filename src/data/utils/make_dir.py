from os import makedirs
from os.path import mkdir


def make_dir(path):
    if '/' in path:
        try:
            makedirs(path)
        except FileExistsError:
            pass
    else:
        try:
            mkdir(path)
        except FileExistsError:
            pass
