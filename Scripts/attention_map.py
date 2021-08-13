# tf-keras-vis
# Install: pip install tf-keras-vis tensorflow
# From: https://github.com/keisen/tf-keras-vis/blob/master/examples/attentions.ipynb

import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import cm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import TrainConfig as config
from Evaluate import get_best_model_path
from Utils import showimg

MODEL_PATH = get_best_model_path()
DATA_FOLDER = config.data_folder + '\\Test'

preprocess_input = lambda x: x / 127.5 - 1.0

# Load model / modify
model = load_model(str(MODEL_PATH))
model.summary()

# Load data
rescale_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
data_gen = rescale_gen.flow_from_directory(DATA_FOLDER, batch_size=1, class_mode='categorical', target_size=config.input_shape[:2])
X, label_one_hot = next(data_gen)
score = CategoricalScore([label_one_hot.argmax()])

showimg(((X[0].mean(axis=-1) + 1) * 127.5).astype(np.uint8))

# --- Saliency ---
saliency = Saliency(model, model_modifier=ReplaceToLinear(), clone=True)
saliency_map = saliency(score, X, smooth_samples=20, smooth_noise=0.20)
showimg(saliency_map[0] * 255, cmap='jet')

# ---- Grad CAM ---
gradcam = GradcamPlusPlus(model, model_modifier=ReplaceToLinear(), clone=True)
cam = gradcam(score, X)
heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
showimg(heatmap, cmap='jet')

# --- ScoreCAM ---
scorecam = Scorecam(model, model_modifier=ReplaceToLinear())
cam = scorecam(score, X, penultimate_layer=-1)
heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
showimg(heatmap, cmap='jet')

_ = 'bp'
