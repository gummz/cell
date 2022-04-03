from os import mkdir, makedirs


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
