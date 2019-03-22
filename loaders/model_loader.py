from models import *

class ModelLoader:
    model = None

    def __init__(self, args):
        self.use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        if args.data_name == 'MNIST':
            self.loadMNISTNet(args)
    
    def loadMNISTNet(self, args):
        self.model = MNISTNet()