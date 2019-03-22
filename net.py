from loaders import *

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

    
    def _build_model(self):
        # Load the model
        _model_loader = ModelLoader(self.args)
        self.model = _model_loader.model

        # If continue_train, load the pre-trained model
        if self.args.continue_train:
            self._load_model()

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
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
    
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

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
    
    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            self._train_epoch(epoch)
            self.test()

    def save_model(self):
        if not os.path.exists(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)
        model_filename = self.args.checkpoint_dir + 'model.pth'
        torch.save(self.model.state_dict(), model_filename)
    
    def _load_model(self):
        if not os.path.exists(self.args.checkpoint_dir):
            print('Checkpoint Directory does not exist. Starting training from epoch 0.')
            return
        # Find the most recent model file
        model_filename = self.args.checkpoint_dir + 'model.pth'
        self.model.load_state_dict(torch.load(model_filename))