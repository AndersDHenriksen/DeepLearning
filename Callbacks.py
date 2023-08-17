import os
import shutil
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from DataGenerator import get_datagen_from_folder


class EpochCheckpoint(Callback):
    def __init__(self, output_dir, every=10, save_best=True, best_limit=1e30, verbose=True):
        # call the parent constructor
        super(Callback, self).__init__()

        # store the base output path for the model, the number of epochs that must pass before the model is serialized
        # to disk and the current epoch value
        self.output_dir = output_dir
        self.every = every
        self.save_best = save_best
        self.verbose = verbose
        self.best_val_loss = best_limit
        self.current_latest = ''
        self.current_best = ''

    def on_epoch_end(self, epoch, logs=None):
        # increment the internal epoch counter and get current validation loss
        epoch += 1
        current_val_loss = (logs and logs.get('val_loss')) or 1e31
        tf.summary.scalar('learning rate', data=self.model.optimizer.lr.numpy(), step=epoch)

        # check to see if the model has been validation loss and should be serialized to disk
        if self.save_best and current_val_loss < self.best_val_loss:
            save_path = os.path.join(self.output_dir, f"best_epoch-{epoch:05d}_val-loss-{current_val_loss:.2f}")
            if self.verbose:
                print(f'\nEpoch {epoch:02d}: Validation loss improved from {self.best_val_loss:.2f} to {current_val_loss:.2f}, saving model to {save_path} ... ', end="")
            save_path = self.save_model(save_path)
            self.delete_model(self.current_best)
            self.best_val_loss = current_val_loss
            self.current_best = save_path
            if self.verbose:
                print("Done.")

        # check to see if the model should be serialized to disk due to schedule
        if epoch % self.every == 0:
            save_path = os.path.join(self.output_dir, f"latest_epoch-{epoch:05d}_val-loss-{current_val_loss:.2f}")
            if self.verbose:
                print(f'\nEpoch {epoch:02d}: Saving model to {save_path} ... ', end="")
            save_path = self.save_model(save_path)
            self.delete_model(self.current_latest)
            self.current_latest = save_path
            if self.verbose:
                print("Done.")

    def save_model(self, model_path):
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(model_path)
        else:
            model_path += '.h5'
            self.model.save(model_path)
        return model_path

    @staticmethod
    def delete_model(model_path):
        if not model_path:
            return
        if os.path.isfile(model_path):
            os.remove(model_path)
        else:
            shutil.rmtree(model_path)


class TestFolder(Callback):
    def __init__(self, config, data_folder):
        super(Callback, self).__init__()
        self.data_gen = get_datagen_from_folder(config, data_folder, preprocess_input=None, do_augment=False)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            # test_metrics = self.model.evaluate(self.data_gen)  # Will not work, will use normal val data instead.
            y_label = self.data_gen.classes[self.data_gen.index_array]
            y_pred = self.model.predict(self.data_gen)
            test_acc = (y_label == y_pred.argmax(axis=1)).mean()
            print(f"test_accuracy: {test_acc:.4}")
            tf.summary.scalar('epoch_accuracy', data=test_acc, step=epoch + 1)
