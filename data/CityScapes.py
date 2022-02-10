import torch
import os
from collections import namedtuple
import cv2
from torch.utils.data import Dataset
import numpy as np
import albumentations as albu


CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id', 'has_instances', 'ignore_in_eval', 'color'])

labels = [
    CityscapesClass('unlabeled', 0, 19, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle', 1, 19, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 19, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi', 3, 19, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static', 4, 19, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic', 5, 19, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground', 6, 19, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking', 9, 19, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track', 10, 19, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail', 14, 19, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge', 15, 19, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel', 16, 19, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup', 18, 19, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan', 29, 19, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer', 30, 19, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate', -1, 19, 'vehicle', 7, False, True, (0, 0, 142)),
]

trainId2name_dic = {label.train_id: label.name for label in labels}
trainId2name = []
for i in range(len(trainId2name_dic)):
    trainId2name.append(trainId2name_dic[i])
#print("\n-------trainId2name-------")
#print(trainId2name)
#for id in trainId2name_dict:
#    trainId2name.append(trainId2name_dict[id])

#print(trainId2name)
class_cityscape={label.train_id: label.name for label in labels}

class_weight_cityscape=torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])

class CityscapesTransform:
    def get_training_augmentation(self):
        train_transform = [
            albu.HorizontalFlip(p=0.5),

            albu.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255),
                             mask_value=0, p=1),

            albu.OneOf(
                [
                    albu.RandomSizedCrop([360, 1024], 640, 800, w2h_ratio=4 / 3, interpolation=1, p=1.0),
                    albu.RandomCrop(height=640, width=800, always_apply=True),
                    # albu.RandomSizedCrop([240, 720], 320, 320, w2h_ratio=4 / 3, interpolation=1, p=1.0),
                    # albu.RandomCrop(height=320, width=320, always_apply=True),
                ],
                p=1.0,
            ),

            albu.GaussNoise(p=0.1),
            albu.Perspective(p=0.1),
        ]

        return albu.Compose(train_transform)


    def get_validation_augmentation(self):
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            albu.PadIfNeeded(1024, 2048),
            #albu.CenterCrop(height=720, width=960, always_apply=True),
            albu.CenterCrop(height=1024, width=2048, always_apply=True),
        ]

        return albu.Compose(test_transform)


    def to_tensor(self,x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')


    def get_preprocessing(self,preprocessing_fn):
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=self.to_tensor, mask=self.to_tensor),
        ]
        return albu.Compose(_transform)


class Cityscapes_Dataset(Dataset):
    def __init__(
            self,
            images_dir,
            targets_dir,
            augmentation=None,
            preprocessing=None,
    ):

        self.images_dir=images_dir
        self.targets_dir=targets_dir
        self.ids = os.listdir(images_dir)
        self.images_fps = []
        self.masks_fps = []

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images_fps.append(os.path.join(img_dir, file_name))
                mask_name=file_name.split('_leftImg8bit')[0]+"_gtFine_labelTrainIds.png"

                self.masks_fps.append(os.path.join(target_dir, mask_name))

                if not os.path.isfile(os.path.join(target_dir, mask_name)):
                    print("ERROR: %s" %(os.path.join(target_dir, mask_name)))

        # convert str names to class values on masks
        self.class_values = [trainId2name.index(cls) for cls in trainId2name]
        #self.class_values = [self._CLASSES.index(cls.lower()) for cls in classes_camvid]
        self.augmentation = augmentation
        self.preprocessing = preprocessing


    def __getitem__(self, index):
        # read data
        image = cv2.imread(self.images_fps[index])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[index], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]

        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, self.images_fps[index]


    def __len__(self) -> int:
        return len(self.images_fps)



import matplotlib.pyplot as plt

def formatting_label_cityscapes(mask):
    palette = {
        0: (68,1,84),
        1: (68,1,84),
        2: (68,1,84),
        3: (68,1,84),
        4: (68,1,84),
        5: (68,1,84),
        #6: (255,192,0),
        6: (68,1,84),
        7: (68,1,84),
        8: (68,1,84),
        9: (68,1,84),
        10:(68,1,84),
        11:(68,1,84),
        12:(68,1,84),
        13:(30, 210, 242),
        14:(30, 210, 242),
        15:(68,1,84),
        16:(68,1,84),
        17:(68,1,84),
        18:(68,1,84),
        19:(68,1,84)
    }
    mask=torch.squeeze(mask)
    rows = mask.size()[1]
    cols = mask.size()[2]
    mask=mask.numpy()

    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            image[i, j] = palette[np.argmax(mask[:,i, j])]
    return image

if __name__ == "__main__":

    transforms_cityscapes=CityscapesTransform()

    dataset=Cityscapes_Dataset(
        images_dir="E:/data/cityscapes/leftImg8bit/train/",
        targets_dir="E:/data/cityscapes/gtFine/train/",
        augmentation=transforms_cityscapes.get_training_augmentation(),
        preprocessing=None,
    )

    print(dataset.__len__())
    class_values = [cls for cls in trainId2name_dic]
    #for i in range(1000):
    #    img, gt=dataset[i]
    #    print(i)
    mask = cv2.imread("E:/data/cityscapes/gtFine/train/tubingen/tubingen_000020_000019_gtFine_labelTrainIds.png", 0)
    print(type(mask))
    # extract certain classes from mask (e.g. cars)
    masks = [(mask == v) for v in class_values]
    gt = np.stack(masks, axis=-1).astype('float')
    #print(gt.shape)
    #img,gt=dataset[2]
    plt.subplot(1,1, 1).axis('off')
    #plt.title("Attention")
    plt.imshow(formatting_label_cityscapes(torch.from_numpy(gt).transpose(0,2).transpose(1,2)))
    plt.show()