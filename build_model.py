import torch.nn as nn
import torchvision.models as models
def build_model(args):
    print('create model:{model_name}'.format(model_name=args.model))
    model=models.__dict__[args.model](pretrained=True)
    model.fc=nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(512,2)
    )
    return model