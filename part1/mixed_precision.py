import torch

x = torch.rand([32, 32]).cuda()
y = torch.rand([32, 32]).cuda().half()

with torch.cuda.amp.autocast():
    a = x + y
    b = x @ y

print(a.dtype)
print(b.dtype)