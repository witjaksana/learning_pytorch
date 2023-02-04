import torch
import torchvision.models as models

# save pretrained
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

# load
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
print(model.eval())