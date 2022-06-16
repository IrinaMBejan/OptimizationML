import os
from collections import namedtuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models import *


def train(model, optimizer, train_loader, epoch_num=-1, tolerance=0.01, path=None, device='cpu', max_nbr_epochs=10_000,
          val_dataloader=None, sam=False, dir_path="checkpoints"):
    """
    Args:
        model (nn.Module) :                             The model to be trained
        optimizer (torch.optim) :                       The optimizer that will be used during the training 
                                                        of the model.
        train_loader (torch.utils.data.DataLoader):     A dataloader containing the training data
        epoch_num (int) :                               If epoch_num == -1, then the model will be trained 
                                                        until the loss reaches 'tolerance' or 'max_nbr_epochs'
                                                        is reached. Else, we train for 'epoch_num' epochs.
        tolerance (float):                              The tolerance that has to be reached by the loss in 
                                                        order to stop the training when 'epoch_num' == -1.
        path (string):                                  The path where the checkpoint is to be saved. No saving
                                                        is done if 'path' is None
        device (string):                                The device on which we run the training ('cpu' or 'cuda')
        max_nbr_epochs (int):                           The maximum number of epochs the model will be trained
                                                        if we are running with 'epoch' == -1. If this number 
                                                        is reached the training is stopped even if the tolerance
                                                        was not achieved.
        val_dataloader (torch.utils.data.DataLoader):   A dataloader containing the validation data. If this is 
                                                        not None, then the validation accuracy will be computed
                                                        and added to the checkpoint before it is saved.
        sam (boolean):                                  Indicates if the SAM algorithm will be used to optimize 
                                                        the model
        
    Return value:
        (nn.Module) The model trained.
    """
    print('new version 2')

    converged = False

    for epoch in range(max_nbr_epochs):
        ### Check if maximal number of iterations is achieved. If epoch_num = -1, 
        ### than we want loss to be smaller than tolerance
        if epoch == epoch_num:
            break

        correct = 0
        iter_losses = []
        begin = datetime.now()
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):

            # using GPU if available
            data, target = data.to(device), target.to(device)

            if sam == True:
                # enable_running_stats(model)
                y_pred = model(data)
                loss = F.cross_entropy(y_pred, target)
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward step
                # disable_running_stats(model)
                y_pred = model(data)
                loss = F.cross_entropy(y_pred, target)
                loss.backward()
                iter_losses.append(loss.item())
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                y_pred = model(data)
                loss = F.cross_entropy(y_pred, target)
                iter_losses.append(loss.item())
                loss.backward()
                optimizer.step()

            # display
            correct += torch.sum(y_pred.argmax(dim=1) == target)
            if (batch_idx + 1) % 100 == 0:
                curr_loss = sum(iter_losses) / len(iter_losses)
                print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t accuracy: {:.2f}%\t Time: {}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader),
                    curr_loss,
                    (correct + 0.0) * 100 / len(train_loader.dataset),
                    datetime.now() - begin
                ),
                    end='')
            elif batch_idx + 1 == len(train_loader):
                curr_loss = sum(iter_losses) / len(iter_losses)
                epoch_accuracy = (correct + 0.0) * 100 / len(train_loader.dataset)
                print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t accuracy: {:.2f}%\t Time: {}'.format(
                    epoch,
                    len(train_loader.dataset),
                    len(train_loader.dataset),
                    100,
                    curr_loss,
                    epoch_accuracy,
                    datetime.now() - begin
                ),
                    end='')

        print('')

        epoch_loss = sum(iter_losses) / len(iter_losses)
        model.train_losses.append(epoch_loss)
        model.train_accuracies.append(epoch_accuracy)

        # computing validation score
        if val_dataloader != None:
            model.eval()
            test_statistics = test(model, val_dataloader, device=device)
            val_loss, val_accuracy = test_statistics['loss'], test_statistics['accuracy']
            model.val_losses.append(val_loss)
            model.val_accuracies.append(val_accuracy.item())
            print(f'\t Test loss: {val_loss} \t Test accuracy: {val_accuracy}')

        print()

        # saving the model
        if path != None and ((converged == False and curr_loss < tolerance) or (converged and (epoch + 1) % 50 == 0)):
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'training_loss': model.train_losses,
                'training_accuracy': model.train_accuracies
            }

            if val_dataloader != None:
                checkpoint['validation_loss'] = model.val_losses
                checkpoint['validation_accuracy'] = model.val_accuracies

            if converged == False and curr_loss < tolerance:
                new_path = f'converged/{path}'
                torch.save(checkpoint, f'{dir_path}/{new_path}.pt')
                converged = True
            if converged and (epoch + 1) % 50 == 0:
                new_path = f'epoch{epoch + 1}/{path}'
                torch.save(checkpoint, f'{dir_path}/{new_path}.pt')
            elif epoch + 1 == max_nbr_epochs:
                new_path = f'epoch{epoch + 1}/{path}'
                torch.save(checkpoint, f'{dir_path}/{new_path}.pt')

        # checking if the loss is small enough to stop
        if epoch_num == -1 and curr_loss < tolerance:
            break

    return model


def test(model, test_loader, device='cpu'):
    """
    Args:
        model (nn.Module) :                             The model to be trained
        test_loader (torch.utils.data.DataLoader):      A dataloader containing the data on which to test the 
                                                        accuracy of the model
        device (string):                                The device on which we run the training ('cpu' or 'cuda')
        
    Return value:
        (dict) A dictionnary containing the loss and the accuracy as computed on the test data.
    """
    losses = []
    correct = 0.0
    for batch_idx, (data, target) in enumerate(test_loader):
        # using GPU, if available
        data, target = data.to(device), target.to(device)

        y_pred = model(data)
        loss = F.cross_entropy(y_pred, target)
        losses.append(loss.item())

        correct += torch.sum(y_pred.argmax(dim=1) == target)

    return {'loss': sum(losses) / len(losses),
            'accuracy': 100.0 * correct / len(test_loader.dataset)}


def get_model(architecture, dataset):
    """
    Args:
        architecture (string) :     The model to be trained
        dataset (string):           A dataloader containing the data on which to test the 
                                    accuracy of the model

    Return value:
        (nn.Module) A CNN model.
    """
    input_channels = 3 if dataset == 'CIFAR10' else 1
    size = 32 if dataset == 'CIFAR10' else 28
    if architecture == 'SimpleBatch':
        return SimpleBatch(input_channels=input_channels, size=size)
    if architecture == 'MiddleBatch':
        return MiddleBatch(input_channels=input_channels, size=size)
    if architecture == 'ComplexBatch':
        return ComplexBatch(input_channels=input_channels, size=size)


def preprocess_data_for_sharpness(train_data, dataset, device):
    # This cell preproccesses data for calculating the sharpness.
    print(f'Preporcessing dataset {dataset} in order to calculate sharpness...')
    begin = datetime.now()

    x = torch.stack([v[0] for v in train_data])
    y = torch.tensor(train_data.targets)

    x, y = x.to(device), y.to(device)
    data = namedtuple('_', 'x y n')(x=x, y=y, n=len(y))

    print(f'Time needed {datetime.now() - begin}')

    return data


def generate_dataframe(dataset, architecture, checkpoint_folder):
    df = pd.DataFrame()

    for epoch in [50, 100, 150, 200, 'converged']:
        if epoch != 'converged':
            directory = f'{checkpoint_folder}/{dataset}/{architecture}/epoch{epoch}/'
        else:
            directory = f'{checkpoint_folder}/{dataset}/{architecture}/converged/'

        for file_name in os.listdir(directory):
            row = {}

            if file_name.endswith('_hessian.pt') or file_name.endswith('_sharpness.pt'):
                continue

            if 'SAM' in file_name:
                row['Minimization'] = 'SAM'
            else:
                row['Minimization'] = 'regular'
            row['Optimizer'] = file_name.replace('SAM_', '').replace('.pt', '').replace('_', ' ')
            checkpoint = torch.load(f'{directory}{file_name}', map_location=torch.device('cpu'))

            row['training_accuracy'] = checkpoint['training_accuracy'][-1].item()
            row['Validation accuracy'] = checkpoint['validation_accuracy'][-1]
            row['training_loss'] = checkpoint['training_loss'][-1]
            row['val_loss'] = checkpoint['validation_loss'][-1]
            row['Generalization gap'] = row['training_accuracy'] - row['Validation accuracy']
            row['epoch'] = epoch

            sharpness_file = file_name.replace('.pt', '_sharpness.pt')
            if os.path.exists(directory + sharpness_file):
                row['Sharpness'] = torch.load(directory + sharpness_file)['sharpness']
            else:
                row['Sharpness'] = np.nan

            df = df.append(row, ignore_index=True)

    return df


def generate_dataframe_per_dataset(dataset, checkpoints_dir="checkpoints"):
    df_simple = generate_dataframe(dataset=dataset, architecture='SimpleBatch', checkpoint_folder=checkpoints_dir)
    df_simple['Architecture'] = 'SimpleBatch'
    df_middle = generate_dataframe(dataset=dataset, architecture='MiddleBatch', checkpoint_folder=checkpoints_dir)
    df_middle['Architecture'] = 'MiddleBatch'
    df_complex = generate_dataframe(dataset=dataset, architecture='ComplexBatch', checkpoint_folder=checkpoints_dir)
    df_complex['Architecture'] = 'ComplexBatch'

    df = pd.concat([df_simple, df_middle, df_complex], axis=0)

    return df
