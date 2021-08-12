# tf-keras-vis
# Install: pip install tf-keras-vis tensorflow
# From: https://github.com/keisen/tf-keras-vis/blob/master/examples/attentions.ipynb

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from matplotlib import cm
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import standardize
from tf_keras_vis.gradcam import Gradcam
import TrainConfig as config
from Evaluate import get_best_model_path

MODEL_PATH = get_best_model_path()
DATA_FOLDER = config.data_folder + '\\Test'

preprocess_input = lambda x: x / 127.5 - 1.0

# Load model / modify
def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear  # Replace softmax with linear. Somehow required.

model = load_model(str(MODEL_PATH))
model.summary()

# Load data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
rescale_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
data_gen = rescale_gen.flow_from_directory(DATA_FOLDER, batch_size=1, class_mode='categorical', target_size=config.input_shape[:2])
image, label_one_hot = next(data_gen)
loss = lambda output: K.mean(output[:, label_one_hot.argmax()])
X = image

# --- Saliency ---
saliency = Saliency(model, model_modifier)
saliency_map = saliency(loss, X, smooth_samples=20)
saliency_map = standardize(saliency_map)

plt.figure()
plt.imshow(((X[0].mean(axis=-1) + 1) * 127.5).astype(np.uint8))
plt.show()

plt.figure()
# plt.imshow((X[0].mean(axis=-1)*255).astype(np.uint8))
plt.imshow(saliency_map[0] * 255, cmap='jet')
plt.waitforbuttonpress()

# --- Grad CAM ---

# Create Gradcam object
gradcam = Gradcam(model, model_modifier)

# Generate heatmap with GradCAM
cam = gradcam(loss, X)
cam = standardize(cam)

plt.figure()
heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
# plt.imshow((X[0].mean(axis=-1)*255).astype(np.uint8))
plt.imshow(heatmap, cmap='jet')
plt.waitforbuttonpress()

_ = 'bp'
