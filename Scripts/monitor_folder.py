from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from Evaluate import get_best_model_path

MONITOR_FOLDER = Path().cwd()
ext = '*.png'

model = load_model(str(get_best_model_path()))
already_inferred = list(MONITOR_FOLDER.glob(ext))
while True:
    for image_path in MONITOR_FOLDER.glob(ext):
        if image_path in already_inferred:
            continue
        already_inferred.append(image_path)
        img = load_img(image_path, target_size=model.input_shape[1:3])
        img_batch = np.array(img)[None, ...].astype(float) / 127.5 - 1.0
        result = np.squeeze(model.predict(img_batch))
        print(f"{image_path}: {result}")
