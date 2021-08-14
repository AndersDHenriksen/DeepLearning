import os
from pathlib import Path
from datetime import datetime
from shutil import copytree, ignore_patterns
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorboard import program
is_matplotlib_available = True
try:
    # noinspection PyUnresolvedReferences
    import matplotlib.pyplot as plt
except ImportError:
    is_matplotlib_available = False


def tensorboard_launch(experiments_folder):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', experiments_folder])
    url = tb.launch()
    print(f'TensorBoard at {url}')


def process_config(copy_code_to_model_dir=True):
    import TrainConfig as config
    config.load_model = None
    experiment_folder = Path(config.experiment_folder)
    do_load_exp = "_run" in config.experiment_name
    if do_load_exp:
        experiment_name = list(experiment_folder.glob('*' + config.experiment_name))
        if len(experiment_name) and len(list((experiment_name[0] / "checkpoint").glob("latest_epoch*"))):
            config.experiment_name = experiment_name[-1].stem
            model_chkpoints = list((experiment_name[0] / "checkpoint").glob("latest_epoch*"))
            config.load_model = sorted(model_chkpoints, key=lambda p: int(p.stem[13:18]))[-1]
            config.model_epoch = int(config.load_model.stem[13:18])
        else:
            do_load_exp = False
    if not do_load_exp:
        run_n = len(list(experiment_folder.glob('*' + config.experiment_name + '*')))
        config.experiment_name = "{} - {}_run{}".format(datetime.now().strftime('%Y-%m-%d %H-%M-%S'), config.experiment_name, run_n)
        config.model_epoch = 0

    config.data_folder = Path(config.data_folder)
    config.log_dir = str(experiment_folder / config.experiment_name / "log") + os.sep
    config.checkpoint_dir = str(experiment_folder / config.experiment_name / "checkpoint") + os.sep

    [Path(dir_path).mkdir(parents=True, exist_ok=True) for dir_path in [config.log_dir, config.checkpoint_dir]]
    tf.summary.create_file_writer(config.log_dir + "metrics").set_as_default()

    if copy_code_to_model_dir:
        code_dir = str(experiment_folder / config.experiment_name / "code") + os.sep
        copytree(src=Path(__file__).parent, dst=code_dir, ignore=ignore_patterns('__pycache__', '.*'), dirs_exist_ok=True)

    if config.disable_gpu:
        print('Disabling GPU! Computations will be done on CPU.')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    return config


def showimg(img, overlay_mask=None, close_on_click=False, title=None, cmap="gray", overlay_cmap="RdBu"):
    if not is_matplotlib_available:
        Image.fromarray(img).show()
        return

    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5, forward=True)
    # Show image
    img = np.squeeze(img)
    if img.ndim == 1:
        plt.plot(img)
    else:
        plt.imshow(img, cmap=cmap)
        if overlay_mask is not None:
            masked = np.ma.masked_where(overlay_mask == 0, overlay_mask)
            plt.imshow(masked, overlay_cmap, alpha=0.5)
    if title:
        plt.title(title)
    # Trim margins
    plt.tight_layout()
    if close_on_click:
        plt.waitforbuttonpress()
        plt.close(fig)
    else:
        plt.show()
    return fig
