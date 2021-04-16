"""Convert the tensorflow model weights from the provided .npy file to a PyTorch checkpoint
by loading them into a PyTorch model and saving the state dictionary of that model."""

import numpy as np
import torch
from torch import nn

from model import Model


def get_model():
    np_weight_dict = np.load('deephand_model.npy', allow_pickle=True, encoding='latin1').item()

    model = Model()
    num_errors = 0
    for key in np_weight_dict.keys():
        if key.startswith('loss1') or key.startswith('loss2'):  # For inference we don't need the auxiliary layers.
            continue
        if key.startswith('loss3'):  # Final classification layer
            source_weight = torch.from_numpy(np_weight_dict[key]['weights']).permute(1, 0)
            source_bias = torch.from_numpy(np_weight_dict[key]['biases'])
        else:  # One of the convolutional layers
            source_weight = torch.from_numpy(np_weight_dict[key]['weights']).permute(3, 2, 1, 0)
            source_bias = torch.from_numpy(np_weight_dict[key]['biases'])
        target_weight = model.__getattr__(key).weight
        target_bias = model.__getattr__(key).bias
        if source_weight.size() != target_weight.size():
            print(f'[{key}] Invalid weight size! Source: {source_weight.size()}. Target: {target_weight.size()}')
            num_errors += 1
        else:
            print(f'[{key}] Weights ok!')
        if source_bias.size() != target_bias.size():
            print(f'[{key}] Invalid bias size! Source: {source_bias.size()}. Target: {target_bias.size()}')
            num_errors += 1
        else:
            print(f'[{key}] Bias ok!')
        model.__getattr__(key).weight = nn.Parameter(source_weight)
        model.__getattr__(key).bias = nn.Parameter(source_bias)
    if num_errors > 0:
        print('Encountered at least one error during conversion. Please check the output above.')
    else:
        print('Conversion succeeded without any errors.')

    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    model = get_model()
    save_model(model, 'deephand.pth')
