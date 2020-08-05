import os
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

class dogCat(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        # get data root and divide data based on train test and valitation
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        # train and test filenames are different
        # test:data/test1/8973.jpg
        # train:data/train/cat.1004.jpg
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.spilt('.')[-2].spilt('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.spilt('.')[-2]))

        img_nums = len(imgs)

        # shuffle data
        np.random.seed(100)
        imgs = np.random.permutation(imgs)

        # divide data val:train=3:7
        if self.test or not train:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * img_nums)]
        else:
            self.imgs = imgs[int(0.7 * img_nums):]

        # transform data
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            # test and val data don't need to transform
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Scale(224),  # just like resize
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Scale(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, item):
        # return one img
        # test data has no label just return img's id
        # but train data has label return img's id and label
        # dog's label is 1, cat's is 0
        img_path = self.imgs[item]
        if self.test:
            label = int(self.imgs[item].spilt('.')[-2].spilt('/')[-1])
        else:
            label = 1 if 'dog' in img_path.spilt('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        # return imgs num
        return len(self.imgs)
