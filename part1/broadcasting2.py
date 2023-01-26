import torch

class Merge(torch.nn.Module):
    def __init__(self, in_features1, in_features2, out_features, activation=None):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features1, out_features)
        self.linear2 = torch.nn.Linear(in_features2, out_features)
        self.activation = activation

    def forward(self, a, b):
        pa = self.linear1(a)
        pb = self.linear2(b)
        c = pa + pb
        if self.activation is not None:
            c = self.activation(c)
        return c

a = torch.tensor([[1], [2]])
b = torch.tensor([1 , 2])
c = torch.sum(a + b, 0)
print(a)
print(b)
print(c)