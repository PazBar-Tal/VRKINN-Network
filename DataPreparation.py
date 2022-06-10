from dataset import PairedRoom
import configuration
import torch
from torch.utils.data import Dataset



def dataPreparation():
    # train data:
    list_paired_train = torch.load(configuration.TRAIN_LIST)
    labels_train = torch.load(configuration.TRAIN_LABELS_LIST)

    data_train = PairedRoom(list_paired_train, labels_train)


    # test data:
    list_paired_test = torch.load(configuration.TEST_LIST)
    labels_test = torch.load(configuration.TEST_LABELS_LIST)

    data_test = PairedRoom(list_paired_test, labels_test)

   # val data:
    list_paired_val = torch.load(configuration.VAL_LIST)
    labels_val = torch.load(configuration.VAL_LABELS_LIST)
    data_val = PairedRoom(list_paired_val, labels_val)

    return data_train, data_test, data_val