import os

# specify the shape of the inputs for our network
ROOM_SHAPE = (2 ,6, 450)
PR_SHAPE = (6,450)

# specify the batch size and number of epochs
BATCH_SIZE = 11
EPOCHS = 100
patience = 30

# list of hyper-parameters:
num_of_filters1 = 34
num_of_filters2 = 15
karnel1 = 3
stride1 = 3
karnel2 = 2
stride2 = 1

drop_out1 = 0.1
drop_out2 = 0.6

output_fc1 = 20
hidden_size_lstm = 3
output_fc2 = 10

# define the path to the base output directory
BASE_OUTPUT = r'C:\Users\User\PycharmProjects\NN\26.5.22'

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

# lists of train, test, validation paths:
TRAIN_LIST = os.path.sep.join([BASE_OUTPUT, "train_list.pt"])
TRAIN_LABELS_LIST = os.path.sep.join([BASE_OUTPUT, "labels_train_list.pt"])
TEST_LIST = os.path.sep.join([BASE_OUTPUT, "test_list.pt"])
TEST_LABELS_LIST = os.path.sep.join([BASE_OUTPUT, "labels_test_list.pt"])
VAL_LIST = os.path.sep.join([BASE_OUTPUT, "val_list.pt"])
VAL_LABELS_LIST = os.path.sep.join([BASE_OUTPUT, "labels_val_list.pt"])

