import torch
class args(object):
    trainroot = 'data/train'
    testroot='data/test'
    train_batchsize = 32
    test_batchsize = 16
    numworkers = 4
    max_epoch = 5
    device='cuda' if torch.cuda.is_available() else 'cpu'
    save_path = 'ckpt/model.pth'