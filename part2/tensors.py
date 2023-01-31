import torch
import numpy as np

# initializing tensors
data = [[1,2], [3,4]]
x_data = torch.tensor(data)

print(x_data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(data)
print(np_array)
print(x_np)

# from another tensor
x_ones = torch.ones_like(x_data)
print(f"ones tensor:\n {x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"random tensor:\n {x_rand}\n")

# attributes
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

if torch.cuda.is_available():
    tensor = tensor.to("cuda")

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:,0]}")
print(f"Last column: {tensor[:, -1]}")
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim = 1)
print(t1)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
print(y1)
print(y2)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out = y3)
print(y3)

# element-wise
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(z1)
torch.mul(tensor, tensor, out=z3)

agg =tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# in-place operation
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Bridge with NumPy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")