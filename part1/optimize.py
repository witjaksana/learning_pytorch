import torch

def batch_gather(tensor, indices):
    output = []
    for i in range(tensor.size(0)):
        output += [tensor[i][indices[i]]]
    return torch.stack(output)

@torch.jit.script
def batch_gather_jit(tensor, indices):
    output = []
    for i in range(tensor.size(0)):
        output += [tensor[i][indices[i]]]
    return torch.stack(output)

def batch_gather_vec(tensor, indices):
    shape = list(tensor.shape)
    flat_first = torch.reshape(
        tensor, [shape[0] * shape[1]] + shape[2:])
    offset = torch.reshape(
        torch.arange(shape[0]).cuda() * shape[1],
        [shape[0]] + [1] * (len(indices.shape) - 1))
    output = flat_first[indices + offset]
    return output