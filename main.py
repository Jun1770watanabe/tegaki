import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import numpy as np

from network import SimpleNetwork
from network import ConvNetwork
import make_data
import tools


def reshape_by_type(x, model, batch_size):
    if type(model) is SimpleNetwork:
        return x.view(-1, 16) # reshape tensor
    elif type(model) is ConvNetwork:
        return x.reshape(batch_size,1,4,4)

def train(dnn, cnn, batch_size, id, unit, od, iterat, log_int, alpha):
    raw_data, raw_lbl = make_data.make_train_data()
    train_data, train_lbl = make_data.Ingenuity(raw_data, raw_lbl, alpha)
    # train_data, train_lbl = make_data.Ingenuity2(train_data, train_lbl, 2, alpha)
    if dnn:
        model = SimpleNetwork(id, unit, od)
    elif cnn:
        model = ConvNetwork(id, unit, od)
    print(model)

    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    ite_list = []
    loss_list = []
    num_td = len(train_data)

    for ite in range(iterat):
        idx = tools.generate_idx(batch_size, num_td)
        x = reshape_by_type(train_data[idx], model, batch_size)
        y_p = model.forward(x)
        # loss = loss_fn(y_p, train_lbl[idx]) # in case of MSELoss
        loss = loss_fn(y_p, torch.argmax(train_lbl[idx], dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        # store the score for plot
        ite_list.append(ite)
        loss_list.append(loss)

        if ite % log_int == 0:
            ep = int(ite / (num_td / batch_size))
            print(f"Iteration {ite}: Epoch {ep}: loss={loss}")

    tools.plot_loss(ite_list, loss_list, dnn, cnn)
    return model

def test(model):
    test_data, test_lbl = make_data.make_test_data()
    for i in range(len(test_data)):
        x_t = reshape_by_type(test_data[i], model, 1)
        y_t = model(x_t)
        print("============ {} ============".format(i+1))
        print("  estimated label :  ", end="")
        print(tools.tensor2scalar(torch.argmax(y_t)))
        print("   correct label  :  ", end="")
        print(tools.tensor2scalar(torch.argmax(test_lbl[i])))
        print("===========================")

def main():
    parser = argparse.ArgumentParser(description='Handwrite character recognition')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='batch_size')
    parser.add_argument('-u', '--unit', default=100, type=int, help='number of unit')
    parser.add_argument('-id', '--input_dim', default=16, type=int, help='number of input dimension')
    parser.add_argument('-od', '--output_dim', default=3, type=int, help='number of output dimension')
    parser.add_argument('-ite', '--iteration', default=500, type=int, help='number of iteration')
    parser.add_argument('-li', '--log_interval', default=2, type=int, help='number of log interval')
    parser.add_argument('-dnn', action="store_true", help="deep neural network")
    parser.add_argument('-cnn', action="store_true", help="convolutional neural network")
    parser.add_argument('-a', '--alpha', default=10, type=int, help='number of iteration about adding data')
    args = parser.parse_args()

    model = train(
        args.dnn, args.cnn,
        args.batch_size, 
        args.input_dim, 
        args.unit, 
        args.output_dim, 
        args.iteration, args.log_interval,
        args.alpha
        )
    test(model)

if __name__ == "__main__":
    main()