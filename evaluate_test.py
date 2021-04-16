import argparse

import torch
from torch.utils.data import DataLoader

from tf.data import TestDataset
from tf.model import Model


def main(test_data_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = TestDataset(test_data_dir)
    model = Model()
    model.load_state_dict(torch.load('deephand.pth'))
    model = model.to(device)
    model.eval()

    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, drop_last=False)

    print(model)

    with torch.no_grad():
        correct = 0
        total = 0
        for _, sample in enumerate(data_loader):
            imgs, labels = sample
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            predicted = torch.argmax(outputs, dim=1)
            for j in range(outputs.size(0)):
                y_pred = predicted[j].item()
                y_true = labels[j].item()

                if y_pred == y_true:
                    correct += 1
                total += 1
        print(f'Accuracy: {correct / total}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('test_data_dir', type=str, help='Path to the test image root directory of 1MioHands')

    main(argparser.parse_args().test_data_dir)
