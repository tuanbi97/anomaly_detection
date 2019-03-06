from network import R2Plus1D_model
import torch
inputs = torch.rand(1, 3, 16, 112, 112)
net = R2Plus1D_model.R2Plus1DClassifier(101, (2, 2, 2, 2), pretrained=False)

outputs = net.forward(inputs)
print(outputs.size())
