from DataGeneratorSegmentation import get_data_for_segmentation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
import tensorflow.keras.backend as K
from Utils import process_config, tensorboard_launch
from Callbacks import EpochCheckpoint
from Evaluate import save_overlay_images
import segmentation_models as sm
from keras_unet.models import custom_unet


def sigmoid_iou_loss(y_true, y_pred):
    return 1 - K.sum(K.minimum(y_true, y_pred)) / K.sum(K.maximum(y_true, y_pred))


# read config JSON file
config = process_config()

# load data
preprocess_input = None  # sm.get_preprocessing('resnet18')
train_gen, validation_gen = get_data_for_segmentation(config, preprocess_input)

# load model or create new
if config.load_model:
    model = load_model(str(config.load_model), compile=False)
else:
    model = custom_unet(input_shape=config.input_shape, use_batch_norm=True, num_classes=1,
                        filters=4, dropout=0.2, output_activation='sigmoid')
    # model = sm.Unet('resnet18', input_shape=config.input_shape, classes=1, activation='sigmoid',
    #                 decoder_filters=(256, 128, 64, 32, 16), encoder_freeze=False, encoder_weights='imagenet')
model.compile(Adam(lr=config.learning_rate), sm.losses.bce_jaccard_loss,
              metrics=['mean_squared_error', sm.metrics.iou_score, sm.losses.bce_jaccard_loss, sigmoid_iou_loss])
model.summary()

# define callbacks. Learning rate decrease, tensorboard etc.
model_checkpoint = EpochCheckpoint(config.checkpoint_dir)
tensorboard = TensorBoard(log_dir=config.log_dir, profile_batch=0)
callbacks = [model_checkpoint, tensorboard]
if config.use_learning_rate_decay:
    learning_rate_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, verbose=1, cooldown=50)
    callbacks.append(learning_rate_decay)

# launch tensorboard
tensorboard_launch(config.experiment_folder)

# train the network
model.fit(train_gen,
          epochs=config.training_epochs,
          callbacks=callbacks,
          validation_data=validation_gen,
          initial_epoch=config.model_epoch)

if input("Enter 1 to save overlay images: ") == "1":
    save_overlay_images(model, validation_gen)
