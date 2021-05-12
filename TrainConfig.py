experiment_name = 'StandardCnn'
training_epochs = 1000
batch_size = 1
input_shape = [224, 224, 3]
test_split_ratio = 0.3
learning_rate = 0.00002
use_learning_rate_decay = True
disable_gpu = False
data_folder = r"D:\NN\Data\DlInspect"
experiment_folder = r"D:\NN\Experiments\DlInspect"

# Transfer learning warmup parameters
training_epochs_warmup = training_epochs // 10
learning_rate_warmup = 10 * learning_rate
