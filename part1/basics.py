import torch
import numpy as np


def tensorinit():
    a = torch.tensor(3)
    b = torch.zeros([2,2])
    c = torch.rand([2,2,2])
    print(a)
    print(b)
    print(c)


def tensormult():
    a = torch.rand([3,5])
    b = torch.rand([5,4])
    print(a)
    print(b)
    c = a @ b
    print(c)


def tensoradd():
    a = torch.rand([3,3])
    b = torch.rand([3,3])
    print(a)
    print(b)

    # add two tensors
    c = a + b
    print(c)


def tensorconvert():
    a = torch.rand([3,3])
    b = a.numpy()
    print(a)
    print(b)

    c = np.random.normal([3,4])
    print(c)
    d = torch.tensor(c)
    print(d)


def main():
    # init tensor
    tensorinit()

    # add tensor
    tensoradd()

    # mult tensor
    tensormult()

    # convert tensor
    tensorconvert()
    

if __name__ == "__main__":
    main()