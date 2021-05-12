from tensorflow.keras.callbacks import Callback
import os


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
        logs.update({'lr': self.model.optimizer.lr.numpy()})  # Add Learning Rate to tensorboard

        # check to see if the model has been validation loss and should be serialized to disk
        if self.save_best and current_val_loss < self.best_val_loss:
            save_path = os.path.join(self.output_dir, f"best_epoch-{epoch:05d}_val-loss-{current_val_loss:.2f}.hdf5")
            if self.verbose:
                print(f'\nEpoch {epoch:02d}: Validation loss improved from {self.best_val_loss:.2f} to {current_val_loss:.2f}, saving model to {save_path} ... ', end="")
            self.model.save(save_path)
            os.remove(self.current_best) if self.current_best else None
            self.best_val_loss = current_val_loss
            self.current_best = save_path
            if self.verbose:
                print("Done.")

        # check to see if the model should be serialized to disk due to schedule
        if epoch % self.every == 0:
            save_path = os.path.join(self.output_dir, f"latest_epoch-{epoch:05d}_val-loss-{current_val_loss:.2f}.hdf5")
            if self.verbose:
                print(f'\nEpoch {epoch:02d}: Saving model to {save_path} ... ', end="")
            self.model.save(save_path)
            os.remove(self.current_latest) if self.current_latest else None
            self.current_latest = save_path
            if self.verbose:
                print("Done.")