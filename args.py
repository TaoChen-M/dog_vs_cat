import torch
import argparse


def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='resnet34',help='model')
    parser.add_argument("--trainroot",type=str,default='data/train',help='data to train')
    parser.add_argument('--testroot',type=str,default='data/test',help='data to test')
    parser.add_argument('--train_batch',type=int,default=32,help='during train each batch')
    parser.add_argument('--test_batch',type=int,default=16,help='during test each batch')
    parser.add_argument('--numworkers',type=int,default=4,help='mul thread load data')
    parser.add_argument('--max_epoch',type=int,default=5,help='max train times')
    parser.add_argument('--device',type=str,default='cuda',help='use device to train, can change to cpu if pc not have GPU')
    parser.add_argument('--save_path',type=str,default='ckpt/model.pth',help='path to save model')
    parser.add_argument('--result_path',type=str,default='ckpt/results.xls',help='test result')
    return parser.parse_args()
