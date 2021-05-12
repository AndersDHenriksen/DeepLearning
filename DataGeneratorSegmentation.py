import random
from pathlib import Path
from shutil import copytree, rmtree
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, save_img, ImageDataGenerator
from tensorflow.keras.utils import Sequence
from DataGenerator import train_test_split, partition_dataset
from Evaluate import add_overlay_to_image
try:
    import imgaug.augmenters as iaa
except ImportError:
    pass


CONE_WIDTH = 100
AUGMENT_DICT_X = {'width_shift_range': 30, 'height_shift_range': 30, 'rotation_range': 360, 'vertical_flip': True,
                  'horizontal_flip': True, 'brightness_range': [0.8, 1.2], 'zoom_range': 0.1}
AUGMENT_DICT_Y = AUGMENT_DICT_X.copy()
AUGMENT_DICT_Y['brightness_range'] = None


def convert_annotation_to_dot_map(annotation_path):
    if not hasattr(convert_annotation_to_dot_map, 'IMG_SIZE'):
        image_path = str(annotation_path).replace('Annotation', 'Input').replace('.npy', '.png')
        convert_annotation_to_dot_map.IMG_SIZE = load_img(image_path).size[::-1]
    img_size = convert_annotation_to_dot_map.IMG_SIZE
    annotations = np.load(annotation_path).astype(np.int)
    dot_map = np.zeros(img_size, bool)
    dot_map[annotations[:, 1], annotations[:, 0]] = True
    return dot_map


def dot_map_to_prop_map(dot_map, convert_to_uint8=True):
    radius = int(CONE_WIDTH // 2.5)
    d = 2 * radius + 1
    X, Y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    R = np.sqrt(X**2 + Y**2)
    cone_pattern = np.clip(1 - R/radius, a_min=0, a_max=1)
    prop_map = np.zeros((dot_map.shape[0] + 2 * radius, dot_map.shape[1] + 2 * radius), np.float32)
    for dot_location in np.argwhere(dot_map):
        prop_map[dot_location[0]:dot_location[0] + d, dot_location[1]:dot_location[1] + d] += cone_pattern
    if not convert_to_uint8:
        return prop_map[radius:-radius, radius:-radius].clip(max=1)
    return np.uint8(255 * prop_map[radius:-radius, radius:-radius].clip(max=1))


def save_prop_map(annotation_path):
    dot_map = convert_annotation_to_dot_map(annotation_path)
    prop_map = dot_map_to_prop_map(dot_map)
    save_img(annotation_path.parent / f"{annotation_path.stem}.png", prop_map[:, :, None])


def prepare_for_segmentation(dataset_folder):
    print(f"Preparing files {dataset_folder} ... ", end="")
    split_folders = ['Train', 'Test', 'Val'] if (dataset_folder / 'Val').exists() else ['Train', 'Test']
    for split_folder in split_folders:
        split_path = dataset_folder / split_folder
        [(split_path / folder_name).mkdir(exist_ok=True) for folder_name in ["Input", "Annotation"]]
        class_dirs = [x for x in split_path.iterdir() if x.is_dir() and x.name not in ["Input", "Annotation"]]

        # Move to input
        for dp in class_dirs:
            copytree(dp, split_path / "Input" / dp.stem, dirs_exist_ok=True)
            rmtree(dp)

        # Move npy files to annotation folder
        for npy_path in split_path.rglob("*.npy"):
            if 'Annotation' in npy_path.parts:
                continue
            new_path = split_path / "Annotation" / npy_path.parent.name / npy_path.name
            new_path.parent.mkdir(parents=True, exist_ok=True)
            npy_path.rename(new_path)
            save_prop_map(new_path)
    print("Done")


class ZipGen(Sequence):
    def __init__(self, gen_x, gen_y):
        self.gen_x, self.gen_y, self.n = gen_x, gen_y, gen_x.n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.gen_x[i], self.gen_y[i]

    def on_epoch_end(self):
        self.gen_x.on_epoch_end()
        self.gen_y.on_epoch_end()


def get_data_for_segmentation(config, preprocess_input=None, augment_validation_data=False):
    target_size = config.input_shape[:2]
    preprocess_x = preprocess_input or (lambda x: x.astype(np.float32) / 127.5 - 1.0)
    preprocess_y = lambda y: y.astype(np.float32) / 255

    # Move to train / test directory
    partition_dataset(config.data_folder, 1 - config.test_split_ratio, 0, config.test_split_ratio)
    prepare_for_segmentation(config.data_folder)
    color_mode = 'grayscale' if config.input_shape[2] == 1 else 'rgb'
    seed = 42

    gens = {}
    for tt in ['Train', 'Test']:
        augment_dict_x = {} if tt == 'Test' and not augment_validation_data else AUGMENT_DICT_X
        augment_dict_y = {} if tt == 'Test' and not augment_validation_data else AUGMENT_DICT_Y
        X_gen = ImageDataGenerator(preprocessing_function=preprocess_x, **augment_dict_x)
        Y_gen = ImageDataGenerator(preprocessing_function=preprocess_y, **augment_dict_y)
        gen_X = X_gen.flow_from_directory(config.data_folder / tt / "Input", target_size, color_mode,
                                          class_mode=None, batch_size=config.batch_size, seed=seed)
        gen_Y = Y_gen.flow_from_directory(config.data_folder / tt / "Annotation", target_size, 'grayscale',
                                          class_mode=None, batch_size=config.batch_size, seed=seed)
        gens[tt] = ZipGen(gen_X, gen_Y)  # Expanded version of zip(gen_X, gen_Y), so steps is specified

    return gens['Train'], gens['Test']


def test_get_data():
    import matplotlib.pyplot as plt
    import TrainConfig as config
    config.data_folder = Path(config.data_folder)
    train_gen, test_gen = get_data_for_segmentation(config)
    for images, masks in train_gen:
        for image, heatmap in zip(images, masks):
            image = np.round((image + 1) * 127.5).astype(np.uint8)
            fig = plt.figure()
            fig.set_size_inches(18, 10, forward=True)
            plt.imshow(add_overlay_to_image(image, heatmap, alpha=1))
            plt.waitforbuttonpress()
            plt.close(fig)


if __name__ == "__main__":
    test_get_data()


# ----------------------------------------------------------------------------------------------------------------------

# def get_all_dot_maps(annotation_folder):
#     return np.array([convert_annotation_to_dot_map(ap) for ap in sorted(Path(annotation_folder).glob("*.npy"))])
#
#
# def get_all_prop_maps(annotation_folder):
#     return np.array([dot_map_to_prop_map(dm) for dm in get_all_dot_maps(annotation_folder)])
#
#
# def get_all_images(image_folder, color_mode):
#     return np.array([np.array(load_img(ip, color_mode=color_mode)) for ip in sorted(Path(image_folder).glob("*.png"))])
#
#
# def get_data(config, preprocessor=None, augment_validation_data=False):
#     preprocessor = preprocessor or (lambda x: x.astype(np.float32) / 127.5 - 1.0)
#     X_all, Y_all = get_all_images(config.data_path), get_all_prop_maps(config.data_path)[..., None]
#     X_all = X_all if X_all.ndim == 4 else X_all[..., None]
#     X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, config.test_split_ratio)
#     seed = 42
#
#     gens = {}
#     for X, Y, do_augment, n in [[X_train, Y_train, True, 'train'], [X_test, Y_test, augment_validation_data, 'test']]:
#         X_gen = ImageDataGenerator(preprocessing_function=preprocessor, **AUGMENT_DICT_X)
#         Y_gen = ImageDataGenerator(**(AUGMENT_DICT_X if augment_validation_data else {}))
#         # X_gen.fit(X_train, augment=True, seed=seed)  # Don't think this is needed when no mean, std scaling
#         # Y_gen.fit(Y_train, augment=True, seed=seed)
#         gen_X = X_gen.flow(X, batch_size=config.batch_size, seed=seed)
#         gen_Y = Y_gen.flow(Y, batch_size=config.batch_size, seed=seed)
#         gens[n] = zip(gen_X, gen_Y)
#     return gens['train'], gens['test']

# -------------------- Below this line is for tf dataset which takes less memory and could be faster -------------------
# Code below could be more maintable with keras image_dataset_from_directory

# class DataGenerator:
#     def __init__(self, config, glob_pattern="*.png"):
#         self.image_paths = list(Path(config.data_path).glob(glob_pattern))
#         self.image_shape = config.input_shape
#         self.color_mode = "grayscale" if config.input_shape[2] == 1 else 'rgb'
#         random.Random(42).shuffle(self.image_paths)
#         self.annotation_paths = [ip.parent / (ip.stem + "_annotation.npy") for ip in self.image_paths]
#         self.n = len(self.image_paths)
#         assert self.n, "DataGenerator could not find any samples"
#
#     def __call__(self):
#         for ip, ap in zip(self.image_paths, self.annotation_paths):
#             X = np.array(load_img(ip, color_mode=self.color_mode))
#             X = X if X.ndim == 3 else X[:, :, None]
#             Y = dot_map_to_prop_map(convert_annotation_to_dot_map(ap, self.image_shape[:2]))[:, :, None]
#             yield X, Y
#
#
# def rescale(img, heatmap):
#     return tf.cast(img, tf.float32) / 127.5 - 1.0, heatmap
#
#
# def get_datasets(config, cache_data=True, use_imgaug=True):
#     data_gen = DataGenerator(config.data_folder)
#     ds = tf.data.Dataset.from_generator(data_gen, (tf.uint8, tf.float32), (config.input_shape, config.input_shape))
#     n_test = round(config.test_split_ratio * data_gen.n)
#     test_dataset = ds.take(n_test)
#     train_dataset = ds.skip(n_test)
#     if cache_data:
#         train_dataset, test_dataset = train_dataset.cache(), test_dataset.cache()
#     else:
#         print("Warning: Not caching dataset in memory, risk of decoding the every images over and over again. Consider caching to file")
#     train_dataset, test_dataset = train_dataset.batch(config.batch_size), test_dataset.batch(config.batch_size)
#     if use_imgaug:
#         train_dataset = train_dataset.map(np_augmentation_wrapper)
#     else:
#         train_dataset = train_dataset.map(tf_augmentation)
#     return train_dataset.map(rescale), test_dataset.map(rescale)
#
#
# def np_augmentation(images, heatmaps):
#     seq = iaa.Sequential([
#         iaa.GaussianBlur((0, 1.0)),
#         iaa.Fliplr(0.5),
#         iaa.Flipud(0.5),
#         iaa.GammaContrast((0.8, 1.2)),
#         iaa.Affine(scale=(0.95, 1.05), rotate=(-180, 180), translate_px={"x": (-50, 50), 'y': (-50, 50)})
#     ])
#     return seq(images=images.numpy(), heatmaps=heatmaps.numpy())
#
#
# def np_augmentation_wrapper(image, mask):
#     image_shape, mask_shape = image.shape, mask.shape
#     [image, mask] = tf.py_function(np_augmentation, [image, mask], [tf.uint8, tf.float32])
#     image.set_shape(image_shape)
#     mask.set_shape(mask_shape)
#     return image, mask
#
#
# def tf_augmentation(images, heatmaps):
#     # First non GT altering transform
#     images = tf.image.random_brightness(images, 0.1)
#     # Ground Truth altering transforms
#     img_and_gt = tf.concat((tf.cast(images, tf.float32), heatmaps), axis=-1)
#     img_and_gt = tf.image.random_flip_left_right(img_and_gt)
#     img_and_gt = tf.image.random_flip_up_down(img_and_gt)
#     images, heatmaps = img_and_gt[:, :, :, :-1], img_and_gt[:, :, :, -1:]
#     return images, heatmaps
#
#
# def benchmark(dataset, num_epochs=5):
#     import time
#     start_time = time.perf_counter()
#     for epoch_num in range(num_epochs):
#         for sample in dataset:
#             time.sleep(0.01)  # Performing a training step
#     tf.print("Execution time:", time.perf_counter() - start_time)
#
#
# def test_pipeline(dataset):
#     import matplotlib.pyplot as plt
#     dataset_np = dataset.as_numpy_iterator()
#     images, heatmaps = dataset_np.next()
#     for image, heatmap in zip(images, heatmaps):
#         fig = plt.figure()
#         fig.set_size_inches(18, 10, forward=True)
#         plt.imshow(add_overlay_to_image(image, heatmap, alpha=1))
#         plt.waitforbuttonpress()
#         plt.close(fig)
#
#
# if __name__ == "__main__":
#     import TrainConfig as config
#     # test_get_data(config)  # Test normal-non-dataset-function
#     train_dataset, test_dataset = get_datasets(config)
#     test_pipeline(train_dataset)
#     benchmark(train_dataset, num_epochs=1)  # Pre run to cache DS
#     benchmark(train_dataset)  # no aug ? sec | np aug ? sec | tf aug ? sec
#     benchmark(train_dataset.prefetch(tf.data.experimental.AUTOTUNE))  # no aug ? sec | np aug ? sec | tf aug ? sec
