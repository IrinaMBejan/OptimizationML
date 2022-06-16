# main.py
import argparse
import os
import sys
from collections import namedtuple
from datetime import datetime

import torch.utils.data.dataloader as dataloader
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST

from optimizers.adashift import AdaShift            # code taken from: https://github.com/MichaelKonobeev/adashift
from optimizers.adabound import AdaBound            # code taken from: https://github.com/Luolc/AdaBound
from optimizers.sam import SAM                      # code taken from: https://github.com/davda54/sam
import argparse

# import functions for calculating sharpness
from sharpness.Minimum import effective as minimum_shaprness_eff        # code taken from: https://github.com/ibayashi-hikaru/minimum-sharpness

from models import *
from helpers import *

TRAIN_BATCH_SIZE = 2**7
VAL_BATCH_SIZE = 1000

def load_data(dataset):
    if dataset == 'CIFAR10':
        #loading datasets
        train_data =  CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(), # ToTensor does min-max normalization.
        ]), )

        test_data = CIFAR10('./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(), # ToTensor does min-max normalization.
        ]), )

        #creating dataLoaders
        train_loader = dataloader.DataLoader(train_data, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
        test_loader = dataloader.DataLoader(test_data, shuffle=False, batch_size=VAL_BATCH_SIZE)

        return train_data, test_data, train_loader, test_loader

    if dataset == 'FashionMNIST':
        #loading datasets
        train_data =  FashionMNIST('./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(), # ToTensor does min-max normalization.
        ]), )

        test_data = FashionMNIST('./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(), # ToTensor does min-max normalization.
        ]), )

        #creating dataLoaders
        train_loader = dataloader.DataLoader(train_data, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
        test_loader = dataloader.DataLoader(test_data, shuffle=False, batch_size=VAL_BATCH_SIZE)

        return train_data, test_data, train_loader, test_loader

    raise Exception("Given dataset name is unknown!")


def get_model_from(architecture, dataset):
    if architecture not in ['SimpleBatch', 'ComplexBatch', 'MiddleBatch'] or dataset not in ['CIFAR10' or 'FashionMNIST']:
        raise Exception("Given model name is unknown!")

    return get_modeL(architecture, dataset)


def get_optimizer(optimizer_name, model, sam=False):
    if optimizer_name == "SGD":
        if sam:
            return SAM(model.parameters(), torch.optim.SGD, lr=0.1)
        return torch.optim.SGD(model.parameters(), lr=0.1)
    elif optimizer_name == "PHB":
        if sam:
            return SAM(model.parameters(), torch.optim.SGD, lr=0.01, momentum=0.8)
        return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    elif optimizer_name == "Adam":
        if sam:
            return SAM(model.parameters(), torch.optim.Adam)
        return torch.optim.Adam(model.parameters())
    elif optimizer_name == "AdaShift":
        if sam:
            return SAM(model.parameters(), AdaShift, lr=0.01)
        return AdaShift(model.parameters(), lr=0.01)
    elif optimizer_name == "AdaBound":
        if sam:
            return SAM(model.parameters(), AdaBound)
        return AdaBound(model.parameters())
    elif optimizer_name == "Adagrad":
        if sam:
            return SAM(model.parameters(), torch.optim.Adagrad)
        return torch.optim.Adagrad(model.parameters())

    raise Exception("Given optimizer name is unknown!")


def compute_sharpness(data, dataset, model, optimizer_name):
    lr = 0.1 if dataset == 'FashionMNIST' else 1
    num_epochs = 100000
    batch_size = 128

    computed = False

    path = os.path.join('checkpoints', optimizer_name, '.pt')
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    while not computed:
        try:
            # Calculating the sharpness. Returns an error if the learning rate is too big
            sharpnesses, losses = minimum_shaprness_eff(data, model, batch_size, lr, num_epochs=num_epochs, optimizer_file=path)

            # storing the sharpness
            sharpness_path = os.path.join('checkpoints', optimizer_name, '_sharpness.pt')
            checkpoint = {'sharpnesses':sharpnesses, 'sharpness':sharpnesses[-1], 'losses': losses}
            torch.save(checkpoint, sharpness_path)

            computed = True
            print(f'Sharpness: {sharpnesses[-1]}')
        except:
            # Error is returned if the learning rate is too big, so in that case learning rate is set to be twice smaller and number of epochs are set to be twice as bigger
            computed = False
            lr /= 2.0
            num_epochs *= 2
            print(f'Use smaller stepsize than {lr}')


def load_data_for_model(dataset, model_arch, optimizer, checkpoints="checkpoints"):
    checkpoint_folder = os.path.join(checkpoints, f'{dataset}/{model_arch}/')

    # Loading information about SGD training
    checkpoint = torch.load(checkpoint_folder + optimizer + '.pt')
    losses_sgd = checkpoint['training_losses']
    acc_sgd = checkpoint['validation_accuracies']
    sharpness_opt = torch.load(checkpoint_folder + optimizer + '_sharpness.pt')['sharpness']

    # Loading information about SAM SGD training
    checkpoint_sam = torch.load(checkpoint_folder + 'sam_' + optimizer + '.pt')
    losses_sam = checkpoint_sam['training_losses']
    acc_sam = checkpoint_sam['validation_accuracies']
    sharpness_sam = torch.load(checkpoint_folder+'sam_' + optimizer + '_sharpness.pt')['sharpness']

    # Plotting both
    fig, ax = plt.subplots(2,1)

    ax[0].loglog(losses_sgd, label=optimizer)
    ax[0].loglog(losses_sam, label='SAM')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Training loss')
    ax[0].legend()

    ax[1].loglog(acc_sgd, label=optimizer)
    ax[1].loglog(acc_sam, label='SAM')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Test accuracy')
    ax[1].legend()

    # Writing sharpness
    print(f'Minimum sharpness of ' + optimizer + ': {sharpness_opt}')
    print(f'Minimum sharpness of SAM: {sharpness_sam}')


def main():
    parser = argparse.ArgumentParser(description='Args description')
    parser.add_argument('type', type=str,
                        help='Type of the run expected. It can be train, compute_sharpness, plot.',
                        default="train")

    # Args for train and compute_sharpness
    parser.add_argument('dataset', type=str,
                        help='Dataset to be run model for. It can be CIFAR10 or FashionMNIST.',
                        default="CIFAR10")

    parser.add_argument('model_arch', type=str,
                        help='Architecture of the model. It can be SimpleBatch, MiddleBatch, ComplexBatch.',
                        default="SimpleBatch")

    parser.add_argument('optimizer', type=str,
                        help='It can be SGD, PHB, AdaShift, Adagrad, Adam, AdaBound.',
                        default="SGD")

    parser.add_argument('sam', type=int,
                        help='Whether the sam optimizer should be used (0 no, 1 yes).',
                        default=False)


    # Args for compute_sharpness only
    parser.add_argument('load_existing', type=int,
                        help='Whether to load existing trains or recompute them (0 no, 1 yes).',
                        default=False)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model_arch, args.dataset).to(device)
    optimizer = get_optimizer(args.optimizer, model, sam=args.sam)
    train_data, test_data, train_loader, test_loader = load_data(args.dataset)
    model_path = f'{args.dataset}/{args.model_arch}/' + ('sam_' if args.sam else '') + args.optimizer.lower()

    if args.type == "train":
        model = train(model, optimizer, train_loader=train_loader, device=device, max_nbr_epochs=3, path=model_path, val_dataloader=test_loader, sam=args.sam)
    elif args.type == "compute_sharpness":
        if not args.load_existing:
            # Retrain to be able to load them from disk
            model = train(model, optimizer, train_loader=train_loader, device=device, max_nbr_epochs=200, path=model_path, val_dataloader=test_loader, sam=args.sam)
        data = preprocess_data_for_sharpness(train_data, args.dataset, device)
        compute_sharpness(data, args.dataset, model, 'sam_' if args.sam else '' + args.optimizer.lower())
    elif args.type == "plot":
        print('For plotting, it is mandatory all runs have been done previously. More exploration can be done in the DataAnalysis notebook.')
        load_data_for_model(args.dataset, args.model_arch, args.optimizer.lower())


if __name__ == "__main__":
    main()
#%%
