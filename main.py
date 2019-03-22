import argparse
from net import Net

parser = argparse.ArgumentParser(description='Arguments for training/testing usig PyTorch')

# Model Training Params
parser.add_argument('--train_batch_size', type=int, default=64,
                    help='Input batch size for training data (default: 64)')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='How many batches to wait before logging training status')
parser.add_argument('--continue_train', type=bool, default=True,
                    help='Continue training of the model')
# Model Testing Params
parser.add_argument('--test_batch_size', type=int, default=1000,
                    help='Input batch size for testing (default: 1000)')
# Setup learning rate and optimization
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='SGD momentum (default: 0.5)')
# Setup device to train
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training')
parser.add_argument('--multi-gpu', action='store_true', default=True,
                    help='Use multiple GPUs if available.')
# Random Seed 
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed to reproduce results(default: 1)')
# Params to save and load models
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--checkpoint_dir', type=str, default='../ckpt/',
                    help='Directory to save models')
# Param to load specific data and DL model
parser.add_argument('--data_name', type=str, default='MNIST',
                    help='Dataset to load and perform training/inference')


def main(args):
    net = Net(args)
    net.train()

    if args.save_model:
        net.save_model()

if __name__=='__main__':         
    args, _ = parser.parse_known_args()
    main(args)