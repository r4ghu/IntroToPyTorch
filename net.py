import os
import glob
from loaders import *
import torch.optim as optim
from tensorboardX import SummaryWriter

class Net(object):
    def __init__(self, args):
        # Args
        self.args = args

        # Setup manual seed
        torch.manual_seed(args.seed)

        # Set the device
        # Select GPU:0 by default
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the data
        print('Load the data...', end='')
        self._build_data_loader()
        print('DONE')

        # Load the model
        print('Load the model...', end='')
        self._build_model()
        print('DONE')

        # Setup Optimizer
        print('Build optimizer...', end='')
        self._build_optimizer()
        print('DONE')

        # Setup summary writer
        self.writer = SummaryWriter('runs/{}'.format(self.args.data_name))

    
    def _build_model(self):
        # Load the model
        _model_loader = ModelLoader(self.args)
        self.model = _model_loader.model

        # If continue_train, load the pre-trained model
        if self.args.phase == 'train':
            if self.args.continue_train:
                self.load_model()

        # If multiple GPUs are available, automatically include DataParallel
        if self.args.multi_gpu and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def _build_data_loader(self):
        _data_loader = DataLoader(self.args)
        self.train_loader = _data_loader.train_loader
        self.test_loader = _data_loader.test_loader

    def _build_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

    def _train_epoch(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.args.iter_count += 1
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.args.log_interval == 0:
                # Add the values to Summary Writer
                self.writer.add_scalar('train/loss', loss.item(), self.args.iter_count)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
            if self.args.iter_count % self.args.save_frequency == 0:
                self.save_model()
    
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        self.writer.add_scalar('test/loss', test_loss, self.args.iter_count)
        self.writer.add_scalar('test/accuracy', 100. * correct / len(self.test_loader.dataset), self.args.iter_count)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
    
    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            self._train_epoch(epoch)
            self.test()
        # Save the model finally
        if self.args.save_model:
            self.save_model()

    def save_model(self):
        if not os.path.exists(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)
        model_filename = self.args.checkpoint_dir + 'model-{}-{}.pth'.format(self.args.data_name, self.args.iter_count)
        torch.save(self.model.state_dict(), model_filename)
    
    def load_model(self, iter_count=None):
        if not os.path.exists(self.args.checkpoint_dir):
            print('Checkpoint Directory does not exist. Starting training from epoch 0.')
            return
        # Find the most recent model file
        if iter_count is None:
            model_files = glob.glob(self.args.checkpoint_dir + '*.pth')
            if len(model_files) == 0:
                print('No model checkpoint files found.')
                return
            model_prefix = self.args.checkpoint_dir + 'model-{}-'.format(self.args.data_name)
            iter_numbers = [int(x[len(model_prefix):-4]) for x in model_files]
            self.args.iter_count = max(iter_numbers)
        else:
            self.args.iter_count = iter_count
        
        model_filename = self.args.checkpoint_dir + 'model-{}-{}.pth'.format(self.args.data_name, self.args.iter_count)
        self.model.load_state_dict(torch.load(model_filename))