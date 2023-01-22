import torch

def u(x):
    return x * x

def g(u):
    return -u


def differential():
    x = torch.tensor(3.0,  requires_grad = True)
    dgdx = torch.autograd.grad( g(u(x)), x)[0]
    print(dgdx)


if __name__ == "__main__":
    differential()