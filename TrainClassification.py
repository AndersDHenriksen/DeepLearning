from Models import standard_cnn as cnn
from Callbacks import EpochCheckpoint
from DataGenerator import get_data_for_classification
from Utils import process_config, tensorboard_launch
from Evaluate import evaluate_user_choice
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau


# Load config file and data
config = process_config()
train_gen, validation_gen = get_data_for_classification(config)

# Load model or create new
if config.load_model:
    model = load_model(str(config.load_model))
    if config.learning_rate:
        model.optimizer.lr = config.learning_rate
else:
    model = cnn(config, train_gen.num_classes)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=config.learning_rate), metrics=['accuracy'])
model.summary()

# Define callbacks
model_checkpoint = EpochCheckpoint(config.checkpoint_dir, best_limit=0.3)
tensorboard = TensorBoard(log_dir=config.log_dir, profile_batch=0)
callbacks = [model_checkpoint, tensorboard]
if config.use_learning_rate_decay:
    learning_rate_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, verbose=1, cooldown=100)
    callbacks.append(learning_rate_decay)

# Launch tensorboard
tensorboard_launch(config.experiment_folder)

# Special warmup training
if getattr(model, 'is_in_warmup', False):
    model.optimizer.lr = config.learning_rate_warmup
    model.fit(train_gen, epochs=config.training_epochs_warmup, callbacks=callbacks, validation_data=validation_gen)
    for layer in model.layers:
        layer.trainable = True
    model.is_in_warmup = False
    model.optimizer.lr = config.learning_rate
    config.model_epoch = config.training_epochs_warmup

# Train the network
model.fit(train_gen,
          epochs=config.training_epochs,
          callbacks=callbacks,
          validation_data=validation_gen,
          initial_epoch=config.model_epoch)

evaluate_user_choice()
