# My added  code
import os
import tensorflow as tf
tf.keras.backend.set_image_data_format('channels_first')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from transformers import ViTImageProcessor, TFViTForImageClassification
from DataGenerator import get_data_for_classification
from Utils import process_config, tensorboard_launch
from Callbacks import EpochCheckpoint


model_id = "google/vit-base-patch16-224-in21k"  # google/vit-base-patch32-384

# Load config file
config = process_config()
if config.input_shape[0] > 3:
    config.input_shape = [config.input_shape[2], config.input_shape[0], config.input_shape[1]]
    print(f"Changed Input shape to channel first: {config.input_shape}")

# Train in mixed-precision float16. Disable if using a GPU that will not benefit from this
fp16 = False
if fp16:
  tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Prepare dataset
processor = ViTImageProcessor.from_pretrained(model_id)
def preprocesser(x):
    return processor(images=x, return_tensors="np")['pixel_values']
tf_train_dataset, tf_eval_dataset = get_data_for_classification(config, preprocesser)
labels = os.listdir(config.data_folder_train)


if config.load_model:
    model = load_model(str(config.load_model))
    if config.learning_rate:
        model.optimizer.lr = config.learning_rate
else:
    # Load pre-trained ViT model
    model = TFViTForImageClassification.from_pretrained(
        model_id,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)})

    # Compile model
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=Adam(learning_rate=config.learning_rate), loss=loss, metrics=['accuracy'])
model.summary()

# Define callbacks
model_checkpoint = EpochCheckpoint(config.checkpoint_dir, best_limit=0.3)
tensorboard = TensorBoard(log_dir=config.log_dir, profile_batch=0)
callbacks = [model_checkpoint, tensorboard]

# Launch tensorboard
tensorboard_launch(config.experiment_folder)

# Train
train_results = model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    callbacks=callbacks,
    epochs=config.training_epochs,
)

# References
# github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer
# philschmid.de/image-classification-huggingface-transformers-keras