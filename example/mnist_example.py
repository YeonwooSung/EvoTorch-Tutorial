import torch
from torch import nn
from torchvision import datasets, transforms
from evotorch.neuroevolution.net import count_parameters
from evotorch.neuroevolution import SupervisedNE
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PandasLogger


class MNIST30K(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # The first convolution uses a 5x5 kernel and has 16 filters
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 5, stride = 1, padding=2)
        # Then max pooling is applied with a kernel size of 2
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        # The second convolution uses a 5x5 kernel and has 32 filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding = 2)
        # Another max pooling is applied with a kernel size of 2
        self.pool2 = nn.MaxPool2d(kernel_size = 2)

        # The authors are unclear about when they apply batchnorm. 
        # As a result, we will apply after the second pool
        self.norm = nn.BatchNorm1d(1568, affine = False)

        # A final linear layer maps outputs to the 10 target classes
        self.out = nn.Linear(1568, 10)

        # All activations are ReLU
        self.act = nn.ReLU()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Apply the first conv + pool
        data = self.pool1(self.act(self.conv1(data)))
        # Apply the second conv + pool
        data = self.pool2(self.act(self.conv2(data)))

        # Apply layer norm
        data = self.norm(data.flatten(start_dim = 1))

        # Flatten and apply the output linear layer
        data = self.out(data)

        return data


network = MNIST30K()
print(f'Network has {count_parameters(network)} parameters')

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('../datasets', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../datasets', train=False, transform=transform)

#--------------------
# train with evotorch

mnist_problem = SupervisedNE(
    train_dataset,  # Using the dataset specified earlier
    MNIST30K,  # Training the MNIST30K module designed earlier
    nn.CrossEntropyLoss(),  # Minimizing CrossEntropyLoss
    minibatch_size = 256,  # With a minibatch size of 256
    common_minibatch = True,  # Always using the same minibatch across all solutions on an actor
    num_actors = 4,  # The total number of CPUs used
    num_gpus_per_actor = 'max',  # Dividing all available GPUs between the 4 actors
    subbatch_size = 50,  # Evaluating solutions in sub-batches of size 50 ensures we won't run out of GPU memory for individual workers
)

searcher = SNES(mnist_problem, stdev_init = 1, popsize = 1000, distributed = True)

stdout_logger = StdOutLogger(searcher, interval = 1)
pandas_logger = PandasLogger(searcher, interval = 1)

searcher.run(2000)

pandas_logger.to_dataframe().mean_eval.plot()


#--------------------
# Eval

net = mnist_problem.parameterize_net(searcher.status['center']).cpu()

loss = torch.nn.CrossEntropyLoss()
net.eval()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, shuffle = False)
test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        output = net(data)
        test_loss += loss(output, target).item() * data.shape[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
test_loss, correct, len(test_loader.dataset),
100. * correct / len(test_loader.dataset)))

mnist_problem.kill_actors()
