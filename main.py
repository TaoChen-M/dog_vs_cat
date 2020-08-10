import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
from data import dogCat
from tqdm import tqdm
from config import args
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import xlwt

def train(**kwargs):
    # step 1 model
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 2)

    model.to(opt.device)

    # step 2 data
    traindata = dogCat(opt.trainroot, train=True)
    valdata = dogCat(opt.trainroot, train=False)

    train_dataloader = DataLoader(traindata, batch_size=opt.train_batchsize, shuffle=True, num_workers=opt.numworkers)
    val_dataloader = DataLoader(valdata, batch_size=opt.test_batchsize, shuffle=True, num_workers=opt.numworkers)

    # step 3 loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # train
    for epoch in range(opt.max_epoch):

        correct = 0
        for i, (data, label) in tqdm(enumerate(train_dataloader)):
            input = data.to(opt.device)
            target = label.to(opt.device)

            output = model(input)
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            pre = output.argmax(dim=1)
            correct += torch.eq(pre, target).sum().float().item()

        train_accuracy = correct / len(train_dataloader.dataset)
        torch.save(model, opt.save_path)
        val_accuracy = val(model, val_dataloader)

        # save best accuracy
        best_acc = []
        best_acc.append(val_accuracy)
        best_acc.sort()
        print('epoch:{epoch},train_accuracy:{train_accuracy},val_accuracy:{val_accuracy},best_acc:{best_acc}'
              .format(epoch=epoch, train_accuracy=train_accuracy * 100, val_accuracy=val_accuracy * 100,
                      best_acc=100 * best_acc[-1]))

        # visual
        writer = SummaryWriter('ckpt/log/')
        writer.add_scalar('train_acc', train_accuracy, epoch)
        writer.add_scalar('vai_acc', val_accuracy, epoch)


def val(model, dataloader):
    # change model to val
    model.eval()
    correct = 0
    for i, (data, label) in tqdm(enumerate(dataloader)):
        input = data.to(opt.device)
        target = label.to(opt.device)
        score = model(input)
        # print(score)
        pre = score.argmax(dim=1)
        # print(pre)
        correct += torch.eq(pre, target).sum().float().item()
    model.train()
    return correct / len(dataloader.dataset)


def test(**kwargs):
    model = torch.load(opt.save_path)
    model.to(opt.device)
    testdata = dogCat(opt.testroot, test=True)
    test_dataloader = DataLoader(testdata, opt.test_batchsize, num_workers=opt.numworkers)
    results = []
    for i, (data, label) in enumerate(test_dataloader):
        input = data.to(opt.device)
        score = model(input)
        probability = score.argmax(dim=1)
        probability=probability.cpu().numpy()
        print(i, probability)
        # probability
        results.append(probability)
        # time.sleep(0.5)
    results = np.concatenate(results)
    print(results)
    return results

def save(**kwargs):
    results=test()
    len=results.__len__()
    xl=xlwt.Workbook()
    sheet=xl.add_sheet('dog_vs_cat',cell_overwrite_ok=True)
    for i in range(0,len):
        sheet.write(i,0,str(results[i]))
        if results[i]==1:
            sheet.write(i,1,'dog')
        else:
            sheet.write(i,1,'cat')
    xl.save('ckpt/results.xls')

if __name__ == '__main__':
    opt = args()
    # train()
    # test()
    save()
