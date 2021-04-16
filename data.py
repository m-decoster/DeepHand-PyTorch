import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _collect_samples(root_path):
    samples = []
    with open('test_labels.csv', 'r') as labels:
        reader = csv.reader(labels, delimiter=' ')
        for row in reader:
            samples.append({
                'path': os.path.join(root_path,
                                     row[0].replace('final_phoenix_noPause_noCompound_lefthandtag_noClean/', '')),
                'label': int(row[1])
            })
    return samples


class TestDataset(Dataset):
    def __init__(self, root_path):
        self.samples = _collect_samples(root_path)
        self.mean_img = np.load('onemilhands_mean.npy').transpose((1, 2, 0))

    def __getitem__(self, item):
        sample = self.samples[item]

        img = cv2.imread(sample['path'])
        img = cv2.resize(img, (227, 227))
        img = img - self.mean_img
        img = img[1:225, 1:225, :].astype(np.float32)

        img = torch.from_numpy(img).permute(2, 1, 0)

        return img, sample['label']

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    dataset = TestDataset('/home/mcdcoste/Documents/research/data/datasets/data/omio_hands/data/test')
    print(dataset[0][0].shape, dataset[0][1])
