import torch

# Test 1
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1],[2]])

#print(a)
#print(b)

c = a + b
#print(c)

# Broadcasting
a = torch.rand([5,3,5])
b = torch.rand([5,1,6])
#print(b)

linear = torch.nn.Linear(11,10)

tiled_b = b.repeat([1,3,1])
#print(tiled_b)

c = torch.cat([a, tiled_b], 2)
#print(c)
d = torch.nn.functional.relu(linear(c))
#print(d)

print(d.shape)

# Broadcasting implicit
a = torch.rand([5,3,5])
b = torch.rand([5,1,6])

linear1 = torch.nn.Linear(5,10)
linear2 = torch.nn.Linear(6,10)

pa = linear1(a)
pb = linear2(b)

d = torch.nn.functional.relu(pa + pb)
print(d.shape)

