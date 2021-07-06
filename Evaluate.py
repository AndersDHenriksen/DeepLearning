from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import save_img
from Utils import showimg


def confusion_matrix(config, model, validation_gen, do_print=True):
    confusion_gen = validation_gen.image_data_generator.flow_from_directory(config.data_folder_test, shuffle=False,
        batch_size=config.batch_size, class_mode=validation_gen.class_mode, target_size=config.input_shape[:2])
    val_pred = model.predict(confusion_gen)
    if validation_gen.class_mode == 'binary':
        val_pred = val_pred > .5
    else:
        val_pred = np.argmax(val_pred, axis=1)
    confusion_matrix = tf.math.confusion_matrix(confusion_gen.classes, val_pred)
    if do_print:
        print(f'Confusion matrix:')
        for label, cm_row in zip(confusion_gen.class_indices.keys(), confusion_matrix):
            print(f'{label}: {cm_row}')
    return confusion_matrix


def show_errors(model, validation_gen):
    old_batch_size, validation_gen.batch_size = validation_gen.batch_size, 1
    key_name_dict = {v: k for k, v in validation_gen.class_indices.items()}
    for idx in range(validation_gen.n):
        x, y_label = validation_gen[idx]
        y_label = y_label[0].argmax()
        y_pred = model.predict(x)[0].argmax()
        if y_pred == y_label:
            continue
        image = ((x[0] + 1) * 127.5).astype(np.uint8)
        showimg(image, close_on_click=True, title=f"Prediction: {key_name_dict[y_pred]}. Correct {key_name_dict[y_label]}.")
    validation_gen.batch_size = old_batch_size


def mask_to_rgb(mask, color="red"):
    assert color in ["red", "green", "blue", "yellow", "magenta", "cyan"]
    mask = np.squeeze(mask)
    assert mask.ndim == 2
    zeros = np.zeros(mask.shape, np.uint8)
    ones = 255 * np.ones(mask.shape, np.uint8)
    if color == "red":
        return np.dstack((ones, zeros, zeros))
    elif color == "green":
        return np.dstack((zeros, ones, zeros))
    elif color == "blue":
        return np.dstack((zeros, zeros, ones))
    elif color == "yellow":
        return np.dstack((ones, ones, zeros))
    elif color == "magenta":
        return np.dstack((ones, zeros, ones))
    elif color == "cyan":
        return np.dstack((zeros, ones, ones))


def add_overlay_to_image(image, mask, alpha=0.5, color=None):
    assert mask.dtype is not np.uint8
    if image.dtype == np.float32:
        image = ((image + 1) * 127.5).astype(np.uint8)
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        image = np.tile(image, (1, 1, 3))
    color_overlay = mask_to_rgb(mask, color or "red")
    alpha_mask = alpha * mask
    overlay_image = alpha_mask * color_overlay + (1 - alpha_mask) * image
    overlay_image = np.round(overlay_image).astype(np.uint8)
    return overlay_image


def save_overlay_images(model, test_gen, color=None):
    for idx in range(test_gen.n):
        x, y_label = test_gen[idx]
        y_pred = model.predict(x)[0]
        overlay_image = add_overlay_to_image(x[0], y_pred, 1, color)
        save_img(f"overlay{idx:03}.png", overlay_image)


def get_best_model_path():
    import TrainConfig as config
    model_folder = sorted(Path(config.experiment_folder).glob(f"*{config.experiment_name}*"))[-1]
    return sorted((model_folder / 'checkpoint').glob('*best*'))[-1]


if __name__ == "__main__":
    from tensorflow.keras.models import load_model
    from DataGenerator import get_data_for_classification
    from Finalize import finalize_tf2_for_ocv
    import TrainConfig as config

    best_model_path = str(get_best_model_path())
    config.data_folder = Path(config.data_folder)
    train_gen, validation_gen = get_data_for_classification(config)
    model = load_model(best_model_path)

    # Eval and finalize
    confusion_matrix(config, model, validation_gen)
    if input("Enter 1 to see errors: ") == "1":
        show_errors(model, validation_gen)
    if input("Enter 1 to convert best model for OpenCV inference") == "1":
        finalize_tf2_for_ocv(best_model_path)
