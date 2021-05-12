# tf-keras-vis
# Install: pip install tf-keras-vis tensorflow
# From: https://github.com/keisen/tf-keras-vis/blob/master/examples/attentions.ipynb

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from matplotlib import cm
from tf_keras_vis.utils import print_gpus
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from tf_keras_vis.gradcam import Gradcam

MODEL_PATH = r"C:\NN\Experiments\DropletDetection\2020-01-29 07-55-29 - MediumNet_run2\checkpoint\best_epoch-00023_val-loss-0.01.hdf5"
DATA_FOLDER = r"D:\2805 RMED Pre-project Droplet Detection Deep Learning\_Cutout\All"

print_gpus()

# Load model / modify
def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear  # Replace softmax with linear. Somehow required.

model = load_model(MODEL_PATH)
model.summary()

# Load data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
rescale_gen = ImageDataGenerator(rescale=1. / 255)
data_gen = rescale_gen.flow_from_directory(DATA_FOLDER, batch_size=1, class_mode='categorical', target_size=[224, 224])
image, label_one_hot = next(data_gen)
loss = lambda output: K.mean(output[:, label_one_hot.argmax()])
X = image

# --- Saliency ---
saliency = Saliency(model, model_modifier)
saliency_map = saliency(loss, X, smooth_samples=20)
saliency_map = normalize(saliency_map)

plt.figure()
plt.imshow((X[0].mean(axis=-1)*255).astype(np.uint8))
plt.imshow(255*saliency_map[0], cmap='jet', alpha=0.5)

# --- Grad CAM ---

# Create Gradcam object
gradcam = Gradcam(model, model_modifier)

# Generate heatmap with GradCAM
cam = gradcam(loss, X)
cam = normalize(cam)

plt.figure()
heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
plt.imshow(X[0])
plt.imshow(heatmap, cmap='jet', alpha=0.5)
