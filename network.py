import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer_1 = torch.nn.Linear(16, 64)
        self.layer_2 = torch.nn.Linear(64, 128)
        self.layer_3 = torch.nn.Linear(128, 256)
        self.layer_4 = torch.nn.Linear(256, 128)
        self.layer_5 = torch.nn.Linear(128, 64)
        self.layer_6 = torch.nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = F.relu(self.layer_5(x))
        x = F.relu(self.layer_6(x))
        return x


traced_net = torch.jit.trace(Net(), torch.randn(1, 16))
directory = "models/"

torch.jit.save(traced_net, directory + 'net.pt')