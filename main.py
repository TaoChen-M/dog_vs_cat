import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
from data import dogCat
from tqdm import tqdm
from args import get_args
from build_model import build_model
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import xlwt


def train(**kwargs):
    # step 1 model
    model = build_model(args)

    model = model.to(args.device)

    # step 2 data
    traindata = dogCat(args.trainroot, train=True)
    valdata = dogCat(args.trainroot, train=False)

    train_dataloader = DataLoader(traindata, batch_size=args.train_batch, shuffle=True, num_workers=args.numworkers)
    val_dataloader = DataLoader(valdata, batch_size=args.test_batch, shuffle=True, num_workers=args.numworkers)

    # step 3 loss and argsimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # train
    for epoch in range(args.max_epoch):

        correct = 0
        for i, (data, label) in tqdm(enumerate(train_dataloader)):
            input = data.to(args.device)
            target = label.to(args.device)

            output = model(input)
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            pre = output.argmax(dim=1)
            correct += torch.eq(pre, target).sum().float().item()

        train_accuracy = correct / len(train_dataloader.dataset)
        torch.save(model, args.save_path)
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
        input = data.to(args.device)
        target = label.to(args.device)
        score = model(input)
        # print(score)
        pre = score.argmax(dim=1)
        # print(pre)
        correct += torch.eq(pre, target).sum().float().item()
    model.train()
    return correct / len(dataloader.dataset)


def test(**kwargs):
    model = torch.load(args.save_path)
    model.to(args.device)
    testdata = dogCat(args.testroot, test=True)
    test_dataloader = DataLoader(testdata, args.test_batch, num_workers=args.numworkers)
    results = []
    for i, (data, label) in enumerate(test_dataloader):
        input = data.to(args.device)
        score = model(input)
        probability = score.argmax(dim=1)
        probability = probability.cpu().numpy()
        print(i, probability)
        # probability
        results.append(probability)
        # time.sleep(0.5)
    results = np.concatenate(results)
    print(results)
    return results


def save(**kwargs):
    results = test()
    len = results.__len__()
    xl = xlwt.Workbook()
    sheet = xl.add_sheet('dog_vs_cat', cell_overwrite_ok=True)
    for i in range(0, len):
        sheet.write(i, 0, str(results[i]))
        if results[i] == 1:
            sheet.write(i, 1, 'dog')
        else:
            sheet.write(i, 1, 'cat')
    xl.save(args.result_path)


if __name__ == '__main__':
    args = get_args()
    train()
    save()
