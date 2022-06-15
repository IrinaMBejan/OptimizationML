#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
from . import gradient_per_example as hacks
import torch
import copy
import datetime


def report(*args):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
          ' '+' '.join(map(str, args)).replace('\n', ''))


# -------------------------------------------------------------------


def effective(inputs, model, batch_size, is_report=False):
    """
    inputs - inputs of data without targets
    model - architecture we use
    """
    model.eval()
    #print("MODEL", len(list(model.named_parameters())))
    # making the copy of the model and setting all parameters to 0
    diagH = copy.deepcopy(model)
    #print("DIAGH": len(list(diagH.named_parameters())))
    for param in diagH.parameters():
        param.data.zero_()

    # triggered by either the forward or backward pass
    hacks.add_hooks(model)

    n = len(inputs)
    for minibatch in torch.split(inputs, batch_size):
        output = model(minibatch)						# := (b,K), K is # of classe    
        # for p*p^T part
        Z = torch.logsumexp(output, dim=1)			# -> (b)
        loss = Z.sum()									# -> (1 
        loss.backward(retain_graph=True)                    #
        # some smart thing they come up with in the paper
        hacks.compute_grad1(model)
        model.zero_grad()  
        for param, diag in zip(model.parameters(), diagH.parameters()):
            # for BatchNormalization layers (when I tried it with networks without BN it worked)
            # Can I do this?
            if 'grad1' not in dir(param):
                #print('No grad1')
                continue
            # print('-'*100)
            grad1 = param.grad1**2 							# := (b,*)
            grad1 = torch.sum(grad1, dim=0) 					# -> (*)
            diag.data -= grad1.detach()
        hacks.clear_backword1(model)
        # for diag(p) part
        prob = torch.softmax(output, dim=1)				# -> (b,K)
        output = output.sum(dim=0)							# -> (K)
        for k, output_k in enumerate(output):
            output_k.backward(retain_graph=True)
            hacks.compute_grad1(model)
            model.zero_grad()
            for param, diag in zip(model.parameters(), diagH.parameters()):
                if 'grad1' not in dir(param):
                    continue
                grad1 = param.grad1**2				        # := (b,*)
                grad1 = grad1.flatten(1, -1)					# -> (b,prod(*))
                grad1 = prob[:, k].flatten() @ grad1 		# -> (prod(*))
                grad1 = grad1.reshape(param.data.shape)		# -> (*)
                diag.data += grad1.detach()
            hacks.clear_backword1(model)
    for param in diagH.parameters():
        # if 'data' not in dir(param):
        #    print('No data')
        #    continue
        param.data /= n
    hacks.remove_hooks(model)
    return diagH 