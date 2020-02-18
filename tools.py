import torch
from matplotlib import pyplot
import random

def generate_idx(batch_size, num_td):
    """ generate index train data and label
    argument: batch_size (int)
              num_td (int)
                - number of train data
    return: index list
    """
    if batch_size < 1:
        print("please set correct number [batch size]")
        exit()
    else:
        return random.sample(range(num_td), k=batch_size)

def dep_label(alpha, raw_lbl):
    for i in range(alpha):
        temp = raw_lbl.clone()
        if i == 0:
            train_lbl = torch.cat([raw_lbl, temp])
        else:
            train_lbl = torch.cat([train_lbl, temp])
    return train_lbl

def tensor2scalar(tensor):
    if tensor == torch.tensor(0):
        return 0
    elif tensor == torch.tensor(1):
        return 1
    elif tensor == torch.tensor(2):
        return 2

def plot_loss(ite, loss, dnn, cnn):
    pyplot.plot(ite, loss, label="loss")

    if dnn:
        pyplot.title("CrossEntropyLoss score (DNN)")
    if cnn:
        pyplot.title("CrossEntropyLoss score (CNN)")
    pyplot.xlabel("iteration")
    pyplot.ylabel("CEL")
    pyplot.grid()
    pyplot.legend()

    pyplot.show()

