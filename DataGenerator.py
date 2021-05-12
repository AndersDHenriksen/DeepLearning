from pathlib import Path
from shutil import copytree
from random import shuffle
from operator import itemgetter
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


AUGMENT_DICT = {'width_shift_range': 30, 'height_shift_range': 30, 'rotation_range': 360, 'vertical_flip': True,
                'horizontal_flip': True, 'brightness_range': [0.8, 1.2], 'zoom_range': 0.1}


def get_data_for_classification(config, preprocess_input=None, augment_validation_data=False):
    target_size = config.input_shape[:2]
    preprocess_input = preprocess_input or (lambda x: x / 127.5 - 1.0)

    # Move to train / test directory
    partition_dataset(config.data_folder, 1 - config.test_split_ratio, 0, config.test_split_ratio)
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


def partition_dataset(dataset_path, train_split=0.7, validation_split=0.15, test_split=0.15):
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
        category_files = set((p.parent, p.stem) for p in sorted(category_folder.glob("*.*")))  # don't separate files
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
            [file_path.rename(dst_folder / file_path.name) for file_path in folder.glob(f"{stem}.*")]
    [folder.rmdir() for folder in entries_to_move if list(folder.iterdir()) == []]
    print('Done')
