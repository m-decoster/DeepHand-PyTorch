# DeepHand-PyTorch

PyTorch implementation of [Deep Hand](https://www-i6.informatik.rwth-aachen.de/~koller/1miohands/) (inference only).
Based on the [TensorFlow implementation](https://github.com/neccam/TF-DeepHand).

## Requirements

- Python 3.8 
- PyTorch 1.8
- NumPy 1.20

## Running inference

You can run inference by using the `evaluate_test.py` script.
It requires a single argument, which is the path to the test folder of the
One Million Hands dataset.

1. Download the One Million Hands test set
2. Run `python evaluate_test.py /path/to/test/`

## Conversion code

I've also included the original NumPy weights for the TensorFlow model and the script
I used to convert the weights to PyTorch. This is in the `convert_weights.py` script.

Because this is only intended for inference, the auxiliary loss weights `loss1`, `loss2`, are not converted to PyTorch.

This script saves the model as a `.pth` PyTorch checkpoint file.
