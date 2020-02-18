import torch
import numpy as np
import tools

def make_train_data():
    data = torch.tensor(
                       [[[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]],
                        [[0,0,1,0], [0,0,1,0], [0,1,0,0], [0,1,0,0]],
                        [[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]],
                        [[0,0,1,0], [0,0,1,0], [0,1,1,0], [1,0,0,0]],

                        [[1,1,1,1], [1,0,0,1], [1,0,0,1], [1,1,1,1]],
                        [[1,1,1,0], [1,0,0,1], [1,0,0,1], [0,1,1,1]],
                        [[0,1,1,0], [1,0,0,1], [1,0,1,0], [0,1,1,0]],
                        [[1,1,0,1], [1,0,0,1], [0,0,0,1], [1,1,1,1]],

                        [[0,1,1,0], [0,0,1,0], [0,1,0,0], [1,1,1,0]],
                        [[1,1,1,0], [0,0,1,0], [0,1,0,0], [0,1,1,1]],
                        [[1,1,1,1], [0,0,1,0], [0,1,0,0], [1,1,1,1]],
                        [[0,1,1,0], [0,0,1,0], [0,1,0,0], [0,1,1,0]]]
                       , dtype=torch.float32)

    lbl = torch.tensor(
                      [[0,1,0],
                       [0,1,0],
                       [0,1,0],
                       [0,1,0],

                       [1,0,0],
                       [1,0,0],
                       [1,0,0],
                       [1,0,0],

                       [0,0,1],
                       [0,0,1],
                       [0,0,1],
                       [0,0,1]]
                      , dtype=torch.float32)
    return data, lbl

def make_test_data():
    data = torch.tensor(
                        [[[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,1,0,0]],
                         [[0,0,0,1], [0,0,1,0], [0,0,1,0], [0,1,0,0]],
                         [[0,1,1,0], [1,0,0,1], [1,0,1,0], [1,1,1,1]],
                         [[0,0,1,0], [1,0,0,1], [1,0,1,0], [1,0,1,0]],
                         [[0,1,1,0], [1,0,0,1], [0,0,1,0], [0,1,1,1]],
                         [[0,1,1,0], [0,0,1,0], [0,1,0,0], [1,1,1,0]]]
                        , dtype=torch.float32)

    lbl = torch.tensor([[0,1,0],
                        [0,1,0],
                        [1,0,0],
                        [1,0,0],
                        [0,0,1],
                        [0,0,1]]
                        , dtype=torch.float32)
    return data, lbl

def Ingenuity2(raw_data, raw_lbl, mask_size, alpha):
    mask_value = float(raw_data.mean())
    _, h, w = raw_data.shape

    for i in range(alpha):
        batch = np.copy(raw_data)
        for b in batch:
            top = np.random.randint(0 - mask_size // 2, high = (h-mask_size//2) + 1)
            left = np.random.randint(0 - mask_size // 2, high = (w-mask_size//2) + 1)
            bottom = top + mask_size
            right = left + mask_size
            if top < 0:
                top = 0
            if left < 0:
                left = 0
            bottom = top + mask_size
            right = left + mask_size
            b[top:bottom, left:right].fill(mask_value)

        if i == 0:
            train_data = torch.cat([raw_data, torch.from_numpy(batch)])
        else:
            train_data = torch.cat([train_data, torch.from_numpy(batch)])
    train_lbl = tools.dep_label(alpha, raw_lbl)
    return train_data, train_lbl

def gaussian_filter(batch):
    sigma = 0.7
    mean = 0
    for b in batch:
        gauss = np.random.normal(mean, sigma, (4,4))
        gauss = torch.from_numpy(gauss).float()
        b += gauss
    return batch

def Ingenuity(raw_data, raw_lbl, alpha):
    for i in range(alpha):
        gauss_data = gaussian_filter(raw_data.clone()) 
        if i == 0:
            train_data = torch.cat([raw_data, gauss_data])
        else:
            train_data = torch.cat([train_data, gauss_data])
    train_lbl = tools.dep_label(alpha, raw_lbl)
    return train_data, train_lbl


