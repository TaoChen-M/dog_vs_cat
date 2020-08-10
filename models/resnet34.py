import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

if __name__=='__main__':
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 2)
    if torch.cuda.is_available():
        model=model.cuda()
    summary(model,(3,224,224))

