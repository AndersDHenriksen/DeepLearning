from pathlib import Path
from functools import lru_cache
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import save_img
from Utils import showimg


@lru_cache()
def evaluate_cached(model, generator):
    results = []
    for idx in range(generator.n):
        x, y_label = generator[idx]
        img_path = generator.filepaths[idx]
        y_label = y_label[0].argmax()
        y_pred = model.predict(x)[0].argmax()
        results.append((img_path, y_label, y_pred))  # could have been a dataclass instead
    return results


def confusion_matrix(model, validation_gen, do_print=True):
    results = evaluate_cached(model, validation_gen)
    predicts = [r[2] for r in results]
    labels = [r[1] for r in results]
    confusion_matrix = tf.math.confusion_matrix(labels, predicts)
    if do_print:
        print(f'Confusion matrix:')
        for label, cm_row in zip(validation_gen.class_indices.keys(), confusion_matrix):
            print(f'{label}: {cm_row}')
    return confusion_matrix


def show_errors(model, validation_gen):
    key_name_dict = {v: k for k, v in validation_gen.class_indices.items()}
    for idx, (img_path, y_label, y_pred) in enumerate(evaluate_cached(model, validation_gen)):
        if y_pred == y_label:
            continue
        x, _ = validation_gen[idx]
        image = ((x[0] + 1) * 127.5).astype(np.uint8)
        showimg(image, close_on_click=True, title=f"Prediction: {key_name_dict[y_pred]}. Correct {key_name_dict[y_label]}.")


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
    best_model_path = sorted((model_folder / 'checkpoint').glob('*best*'))[-1]
    print(f"Best model path: {best_model_path}")
    return best_model_path


def evaluate_on_generator(model, generator):
    for img_path, y_label, y_pred in evaluate_cached(model, generator):
        print(f"{img_path}, {y_label}, {y_pred}")


def evaluate_user_choice():
    from tensorflow.keras.models import load_model
    from DataGenerator import get_data_for_classification
    from Finalize import finalize_tf2_for_ocv
    import TrainConfig as config

    best_model_path = str(get_best_model_path())
    config.data_folder = Path(config.data_folder)
    train_gen, test_gen = get_data_for_classification(config)
    test_gen = test_gen.image_data_generator.flow_from_directory(config.data_folder_test, shuffle=False,
             batch_size=1, color_mode=test_gen.color_mode, target_size=config.input_shape[:2])
    model = load_model(best_model_path)

    # Eval and finalize
    if input("Enter 1 to calculate confusion matrix: ") == "1":
        confusion_matrix(model, test_gen)
    if input("Enter 1 to print result on validation data: ") == "1":
        evaluate_on_generator(model, test_gen)
    if input("Enter 1 to see errors: ") == "1":
        show_errors(model, test_gen)
    if input("Enter 1 to convert best model for OpenCV inference: ") == "1":
        finalize_tf2_for_ocv(best_model_path)


if __name__ == "__main__":
    evaluate_user_choice()
