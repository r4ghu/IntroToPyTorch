import torch
from torchvision import datasets, transforms

class DataLoader:
    train_loader, test_loader = None, None

    def __init__(self, args):
        self.use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        if args.data_name == 'MNIST':
            self.loadMNIST(args)
    
    def loadMNIST(self, args):
        self.train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', 
                                                        train=True, download=True,
                                                        transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))
                                                        ])), batch_size=args.batch_size, shuffle=True, **self.kwargs)
        
        self.test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', 
                                                        train=False, 
                                                        transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))
                                                        ])), batch_size=args.test_batch_size, shuffle=True, **self.kwargs)
        