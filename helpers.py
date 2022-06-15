import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from models import *


def train(model, optimizer, train_loader, epoch_num=-1, tolerance=0.01, path=None, device='cpu', max_nbr_epochs=10_000,
          val_dataloader=None, sam=False):
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
                ind = path.rindex('/')
                new_path = f'{path[:ind]}/converged{path[ind:]}'
                torch.save(checkpoint, f'checkpoints/{new_path}.pt')
                converged = True
            if converged and (epoch + 1) % 50 == 0:
                ind = path.rindex('/')
                new_path = f'{path[:ind]}/epoch{epoch + 1}{path[ind:]}'
                torch.save(checkpoint, f'checkpoints/{new_path}.pt')
            elif epoch + 1 == max_nbr_epochs:
                ind = path.rindex('/')
                new_path = f'{path[:ind]}/epoch{epoch + 1}{path[ind:]}'
                torch.save(checkpoint, f'checkpoints/{new_path}.pt')

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
