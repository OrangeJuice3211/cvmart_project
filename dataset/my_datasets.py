import numpy as np
import torchvision.transforms.functional as tf
import random
from torch.utils import data
from torchvision import transforms
from PIL import Image
import torch


class MyValDataSet_seg(data.Dataset):
    def __init__(self, root_path, list_path, root_path_coarsemask=None, crop_size=(224, 224)):
        self.root_path = root_path
        self.root_path_coarsemask = root_path_coarsemask
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.files = []
        for name in self.img_ids:
            img_file = name[0:name.find(' ')]
            label_file = name[name.find(' ')+1:]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(self.root_path + datafiles["img"])
        label = Image.open(self.root_path + datafiles["label"])

        image = image.resize((self.crop_h, self.crop_w), Image.BICUBIC)
        label = label.resize((self.crop_h, self.crop_w), Image.NEAREST)

        image = np.array(image) / 255.
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32)

        label = np.array(label)

        name = datafiles["img"][4:]

        return image.copy(), label.copy(), name


# ################ Dataset for Seg
class MyDataSet_seg(data.Dataset):
    def __init__(self, root_path, list_path, root_path_coarsemask=None, crop_size=(224, 224), max_iters=None):
        self.root_path = root_path
        self.root_path_coarsemask = root_path_coarsemask
        self.list_path = list_path
        self.crop_w, self.crop_h = crop_size

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = name[0:name.find(' ')]
            label_file = name[name.find(' ')+1:]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        self.train_augmentation = transforms.Compose(
            [
             transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(512)
             ])

        self.train_gt_augmentation = transforms.Compose(
            [
             transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(512)
             ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(self.root_path + datafiles["img"])
        label = Image.open(self.root_path + datafiles["label"])

        is_crop = [0, 1]
        random.shuffle(is_crop)

        if is_crop[0] == 0:
            [WW, HH] = image.size
            p_center = [int(WW / 2), int(HH / 2)]
            crop_num = np.array(range(30, int(np.mean(p_center) / 2), 30))

            random.shuffle(crop_num)
            crop_p = crop_num[0]
            rectangle = (crop_p, crop_p, WW - crop_p, HH - crop_p)
            image = image.crop(rectangle)
            label = label.crop(rectangle)

            image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
            label = label.resize((self.crop_w, self.crop_h), Image.NEAREST)

        else:
            image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
            label = label.resize((self.crop_w, self.crop_h), Image.NEAREST)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        image = self.train_augmentation(image)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        label = self.train_gt_augmentation(label)

        image = np.array(image) / 255.
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)

        label = np.array(label)
        label = np.float32(label > 0)

        name = datafiles["img"][4:]

        return image.copy(), label.copy(), name


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data_train_root = '/home/data/305/'
    data_train_list = './dataset/kuaidi/Training_seg.txt'

    trainloader = data.DataLoader(MyDataSet_seg(data_train_root, data_train_list, crop_size=(512, 512)),
                                  batch_size=4, shuffle=False, num_workers=4, pin_memory=False, drop_last=True)
    print(len(trainloader))
    for i_iter, batch in enumerate(trainloader):
        images, labels, name = batch
        print(i_iter, name, images.shape, labels.shape, images.max(), labels.max(), labels.min())

        # # plot
        # fig = plt.figure()
        # ax = fig.add_subplot(131)
        # ax.imshow(images[0].cpu().data.numpy().transpose(1, 2, 0))
        # ax.axis('off')
        # ax = fig.add_subplot(132)
        # ax.imshow(labels[0].cpu().data.numpy())
        # ax.axis('off')
        # fig.suptitle('RGB image,ground truth mask', fontsize=6)
        # fig.savefig(name[0] + str(i_iter) + '.png', dpi=200, bbox_inches='tight')
        # ax.cla()
        # fig.clf()
        # plt.close()
        #
        # break













