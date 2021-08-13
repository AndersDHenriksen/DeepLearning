import gc
import random
from pathlib import Path
from shutil import copytree
from random import shuffle
from operator import itemgetter
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
try:
    import imgaug.augmenters as iaa  # conda install -c conda-forge imgaug
except ImportError:
    pass


AUGMENT_DICT = {'width_shift_range': 30, 'height_shift_range': 30, 'rotation_range': 360, 'vertical_flip': True,
                'horizontal_flip': True, 'brightness_range': [0.8, 1.2], 'zoom_range': 0.1}


def get_data_for_classification(config, preprocess_input=None, augment_validation_data=False, timestamp_as_unit=False):
    target_size = config.input_shape[:2]
    preprocess_input = preprocess_input or (lambda x: x / 127.5 - 1.0)

    # Move to train / test directory
    partition_dataset(config.data_folder, timestamp_as_unit, 1 - config.test_split_ratio, 0, config.test_split_ratio)
    config.data_folder_train = str(config.data_folder / 'Train')
    config.data_folder_test = str(config.data_folder / 'Test')

    # Load data generators
    aug_gen = ImageDataGenerator(preprocessing_function=preprocess_input, **AUGMENT_DICT)
    val_gen = aug_gen if augment_validation_data else ImageDataGenerator(preprocessing_function=preprocess_input)

    gens = {}
    color_mode = "grayscale" if config.input_shape[2] == 1 else 'rgb'
    for gen, tt in zip([aug_gen, val_gen], ['Train', 'Test']):
        gens[tt] = gen.flow_from_directory(config.data_folder / tt, batch_size=config.batch_size,
                                           target_size=target_size, color_mode=color_mode)

    return gens['Train'], gens['Test']


def oversample_image_generator(image_generator, n_samples=None):
    current_classes = image_generator.classes
    current_filepaths = np.array(image_generator.filepaths)
    class_count = np.bincount(current_classes)
    n_samples = n_samples or class_count.max()
    new_classes, new_filepaths = [], []
    for class_idx in range(class_count.size):
        n_add = n_samples - class_count[class_idx]
        if n_add <= 0:
            continue
        new_classes += n_add * [class_idx]
        n_repeats = n_add // class_count[class_idx] + 1
        new_filepaths += np.tile(current_filepaths[current_classes == class_idx], n_repeats)[:n_add].tolist()
    image_generator.classes = np.concatenate((current_classes, new_classes))
    image_generator._filepaths += new_filepaths
    image_generator.n = image_generator.samples = image_generator.classes.size
    image_generator._set_index_array()


def train_test_split(X, y, test_split_ratio, random_state=0):
    # Could also be done by sklearn
    rng = np.random.RandomState(random_state)
    indices = np.arange(X.shape[0] if isinstance(X, np.ndarray) else len(X))
    rng.shuffle(indices)
    n_split = int(test_split_ratio * indices.size)
    train_indices, test_indices = indices[n_split:], indices[:n_split]
    if isinstance(X, np.ndarray):
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    return tuple(itemgetter(*indices)(data) for data in [X, y] for indices in [train_indices, test_indices])


def partition_dataset(dataset_path, timestamp_as_unit=False, train_split=0.7, validation_split=0.15, test_split=0.15):
    print(f'Partitioning files from {dataset_path} ... ', end='')
    if train_split + validation_split + test_split != 1:
        print('Also rescaling split fractions to sum to 1 ...', end='')
        train_split, validation_split, test_split = (split/(train_split + validation_split + test_split) for split in
                                                     [train_split, validation_split, test_split])

    dataset_path = Path(dataset_path)
    dataset_path_all = dataset_path / "All"

    assert 'All' != dataset_path.name, "Partition will not work when data is already in an 'All' folder"
    # Create new folders
    new_folders = ["All", "Test", "Train", "Val"] if validation_split else ["All", "Test", "Train"]
    [(dataset_path / new_dir).mkdir(exist_ok=True) for new_dir in new_folders]

    # Rename _annotations.npy
    for anno_path in dataset_path.rglob("*_annotation.npy"):
        anno_path.rename(anno_path.parent / (anno_path.stem[:-11] + ".npy"))

    # Move original files to All folder
    entries_to_move = [dp for dp in dataset_path.glob('*') if (dp.is_dir() and dp.name not in new_folders)]
    [copytree(dp, dataset_path_all / dp.name, dirs_exist_ok=True) for dp in entries_to_move]

    # Get all files, then shuffle while keeping stratified
    partition_paths = [[], [], []]
    for category_folder in entries_to_move:
        if timestamp_as_unit:
            category_files = set((p.parent, p.stem[:19]) for p in sorted(category_folder.glob("*.*")))  # don't separate same timestamp
        else:
            category_files = set((p.parent, p.stem) for p in sorted(category_folder.glob("*.*")))  # don't separate same filename
        category_files = list(category_files)
        shuffle(category_files)
        split_idx = [int(train_split * len(category_files)), int((train_split + validation_split) * len(category_files))]
        partition_paths[0] += category_files[split_idx[1]:]
        partition_paths[1] += category_files[:split_idx[0]]
        partition_paths[2] += category_files[split_idx[0]:split_idx[1]]

    for paths, name in zip(partition_paths, new_folders[1:]):
        for (folder, stem) in paths:
            dst_folder = dataset_path / name / folder.name
            dst_folder.mkdir(exist_ok=True, parents=True)
            [file_path.rename(dst_folder / file_path.name) for file_path in folder.glob(f"{stem}{'' if timestamp_as_unit else '.'}*")]
    [folder.rmdir() for folder in entries_to_move if list(folder.iterdir()) == []]
    print('Done')


# --------------------------------- CODE BELOW HERE IS FOR TFDS DATA GENERATOR -----------------------------------------


def get_data_for_classification_tfds(config, preprocess_input=None, augment_validation_data=False,
                                     timestamp_as_unit=False, aug_method='keras', cache_data=True):
    assert aug_method in ['tf', 'keras', 'imgaug', 'none']

    # Move to train / test directory
    partition_dataset(config.data_folder, timestamp_as_unit, 1 - config.test_split_ratio, 0, config.test_split_ratio)
    config.data_folder_train = str(config.data_folder / 'Train')
    config.data_folder_test = str(config.data_folder / 'Test')

    preprocess_input = preprocess_input or rescale
    gens = {}
    for name, data_folder in zip(['Train', 'Test'], [config.data_folder_train, config.data_folder_test]):
        data_gen = DataGenerator(data_folder, config.input_shape)
        ds = tf.data.Dataset.from_generator(data_gen, (tf.uint8, tf.float32), (config.input_shape, [data_gen.n_labels]))
        if cache_data:
            ds = ds.cache()
        else:
            print("Warning: Not caching dataset in memory, risk of decoding the every images over and over again. Consider caching to file")
        ds = ds.batch(config.batch_size)
        if name == 'Train' or augment_validation_data:
            if aug_method == 'keras':
                ds = ds.map(keras_augmentation_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            elif aug_method == 'imgaug':
                ds = ds.map(imgaug_augmentation_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            elif aug_method == 'tf':
                ds = ds.map(tf_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.map(preprocess_input)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds.num_classes = data_gen.n_labels
        ds.filepaths = data_gen.image_paths
        gens[name] = ds
    return gens['Train'], gens['Test']


class DataGenerator:
    def __init__(self, data_folder, input_shape, glob_pattern="*.png"):
        self.image_paths = list(Path(data_folder).rglob(glob_pattern))
        label_folders = sorted(f.name for f in Path(data_folder).glob("*") if f.is_dir())
        self.input_shape = input_shape
        self.label_int = {f: i for i, f in enumerate(label_folders)}
        self.n_labels = len(label_folders)
        self.color_mode = "grayscale" if input_shape[2] == 1 else 'rgb'
        random.Random(42).shuffle(self.image_paths)
        self.n = len(self.image_paths)
        assert self.n, "DataGenerator could not find any samples"

    def __call__(self):
        for ip in self.image_paths:
            img = load_img(ip, target_size=self.input_shape, color_mode=self.color_mode)
            X = np.array(img)
            X = X if X.ndim == 3 else X[:, :, None]
            Y = self.label_int[ip.parent.name]
            Y = tf.keras.utils.to_categorical(Y, self.n_labels)
            yield X, Y


def rescale(img, label):
    return tf.cast(img, tf.float32) / 127.5 - 1.0, label


def keras_augmentation(images):
    if not hasattr(keras_augmentation, 'aug_gen'):
        keras_augmentation.aug_gen = ImageDataGenerator(**AUGMENT_DICT)
    images = keras_augmentation.aug_gen.flow(images, batch_size=images.shape[0], shuffle=False)[0]
    gc.collect()  # prevent memory build-up for some reason
    return images


def imgaug_augmentation(images):
    seq = iaa.Sequential([
        iaa.GaussianBlur((0, 3.0)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.GammaContrast((0.8, 1.2)),
        iaa.Affine(scale=(0.95, 1.05), rotate=(-180, 180), translate_px={"x": (-30, 30), 'y': (-30, 30)})
    ])
    return seq(images=images.numpy())


def keras_augmentation_wrapper(image, labels):
    image_aug = tf.py_function(keras_augmentation, [image], tf.uint8)
    image_aug.set_shape(image.shape)
    return image_aug, labels


def imgaug_augmentation_wrapper(image, labels):
    image_aug = tf.py_function(imgaug_augmentation, [image], tf.uint8)
    image_aug.set_shape(image.shape)
    return image_aug, labels


def tf_augmentation(images, labels):
    images = tf.cast(images, tf.float32)
    images = tf.image.random_brightness(images, 100)
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    return images, labels


def benchmark(dataset, num_epochs=5):
    import time
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            time.sleep(0.01)  # Performing a training step
    tf.print("Execution time:", time.perf_counter() - start_time)


def test_pipeline(dataset):
    import matplotlib.pyplot as plt
    dataset_np = dataset.as_numpy_iterator()
    images, labels = dataset_np.next()
    for image, label in zip(images, labels):
        fig = plt.figure()
        fig.set_size_inches(18, 10, forward=True)
        plt.imshow(((image + 1) * 127.5).clip(min=0,max=255).astype(np.uint8))
        plt.title(f"Label: {label}")
        plt.waitforbuttonpress()
        plt.close(fig)


if __name__ == "__main__":
    import TrainConfig as config
    config.data_folder = Path(config.data_folder)
    dateset, ds_test = get_data_for_classification_tfds(config)
    test_pipeline(dateset)
    test_pipeline(ds_test)
    benchmark(dateset, num_epochs=1)  # Pre run to cache DS
    benchmark(dateset)
