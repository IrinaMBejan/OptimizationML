#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
# not used anymore
import copy
from . import Hessian_diag as Hessian_diag
import sys
import os
import math
import torch
import datetime


def report(*args):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
          ' '+' '.join(map(str, args)).replace('\n', ''))


# -------------------------------------------------------------------
# loss function that we minimize


class LossFunctionOverAlpha(torch.nn.Module):
    def __init__(self, num_of_layer):
        super().__init__()
        self.exponent = torch.nn.Parameter(torch.zeros(
            num_of_layer-1).double())            # represents alphas
        self.L = num_of_layer-1

    # to be understood better
    def forward(self, weights, biases):
        loss = 0
        alpha = torch.exp(self.exponent)
        prod_scale = 1
        for l in range(self.L):
            loss += weights[l]/(alpha[l]**2)
            prod_scale = prod_scale*alpha[l]
            loss += biases[l]/(prod_scale**2)
        loss += weights[self.L]*(prod_scale**2)
        loss += biases[self.L]
        return loss


def _get_sharpness(weights, biases, lr=1, num_epoch=5, decay=1e-3, is_report=True):
    """
    weights and biases are preprocessed
    """
    # normalize
    scale = sum(weights.values())+sum(biases.values())
    for k in weights:
        weights[k] /= scale
    for k in biases:
        biases[k] /= scale

    # Make a model
    num_of_layer = max(len(weights), len(biases))
    #print(f'Number of layers: {num_of_layer}')
    model = LossFunctionOverAlpha(num_of_layer).double()


#	optimizer = torch.optim.Adam(model.parameters(), lr)	# best lr=1e-1
    # best lr=1e-0, faster than Adam?
    optimizer = torch.optim.SGD(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: 1/(e*decay+1))  # no idea what is it

    begin = datetime.datetime.now()
    losses = [0 for _ in range(num_epoch // 1000)]
    sharpnesses = [0 for _ in range(num_epoch // 1000)]

    # make connection with this code and the paper
    for epoch in range(num_epoch):
        loss = model(weights, biases)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss = loss.item()
        assert not math.isnan(loss), 'nan found, use smaller steps size'
        if is_report and (epoch+1) % 1000 == 0:
            losses[(epoch+1)//1000 - 1] = loss
            sharpnesses[(epoch+1)//1000 - 1] = loss * scale
            print(f'\r \t\t epoch:{epoch+1}\t processed {(epoch+1.0)/num_epoch*100}%\t loss:{loss} \t minimum sharpness: {loss*scale} \t Time needed {datetime.datetime.now()-begin}', end='')
    if is_report:
        print('')
        print('-'*50)
    return sharpnesses, losses


# -------------------------------------------------------------------


def effective(data, model, batch_size, lr, num_epochs, optimizer_file=None):
    """
        data - original data
        model - architecture we use
        batch size, lr, num_
    """

    begin = datetime.datetime.now()
    # model that represents Hessian (not sure)
    
    # check if file already exists
    hessian_file = optimizer_file.replace('.pt', '_hessian.pt')
    if optimizer_file != None and os.path.exists(hessian_file):
        print('\t Loading Hessian')
        diagH = torch.load(hessian_file)
    else:
        print('\t Calculating Hessian')
        diagH = Hessian_diag.effective(data.x, model, batch_size)
        if optimizer_file != None:
            torch.save(diagH, hessian_file)

    print(
        f'\tFinished the diag calculation. Time needed: {datetime.datetime.now()-begin}. Computing sharpness...')

    weights, biases = {}, {}
    l = 0               # iterator
    #print(f'What you can do with diagH', type(diagH))
    for name, param in diagH.named_parameters():
        #print(f'Name of the parameter: {name}\t type {type(name)}')
        # removed in order to be compatible with sequentials
        # if not name.startswith('layer'):
        #    continue
        name = name.replace('layer', '')
        if name.endswith('.weight'):  # layer[i].weight
            name = name.replace('.weight', '')
            #l = int(name)
            weights[l] = param.sum().item()
        if name.endswith('.bias'):		# layer[i].bias
            name = name.replace('.bias', '')
            #l = int(name)
            biases[l] = param.sum().item()
            l += 1

    #print(f'Weights {weights}')
    #print(f'Biases {biases}')
    return _get_sharpness(weights, biases, lr, num_epoch=num_epochs)


#
