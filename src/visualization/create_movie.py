import src.visualization.plot as plot
import src.data.constants as c
import src.data.utils.utils as utils
import os.path as osp
import os

if __name__ == '__main__':
    batch_idx = 1
    # from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    utils.set_cwd(__file__)
    save = osp.join(c.PROJECT_DATA_DIR, 'pred', 'eval', 'track_2D')
    files = os.listdir(save)
    file = files[0]
    load_loc = osp.join(save, file, 'with_tracking', f'batch_{batch_idx}')
    plot.create_movie(load_loc)
