import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as albu

classes_camvid = ['sky', 'building', 'pole', 'road', 'pavement',
           'tree', 'signsymbol', 'fence', 'car',
           'pedestrian', 'bicyclist', 'unlabelled']

#class_weight_camvid = torch.tensor([0.58872014284134, 0.51052379608154, 2.6966278553009, 0.45021694898605, 1.1785038709641,                0.77028578519821, 2.4782588481903, 2.5273461341858, 1.0122526884079, 3.2375309467316,                4.1312313079834, 0])

class_weight_camvid=torch.tensor([0.25652730831033116,0.18528266685745448,4.396287575365375,0.1368693220338383,0.9184731310542199,0.38731986379829597,3.5330742906141994,1.8126852672146507,0.7246197983929721,5.855012980845159,8.136508447439535,1.0974099206087582])
#class_weight_camvid=torch.tensor([0.25652730831033116,0.18528266685745448,4.396287575365375,0.1368693220338383,0.9184731310542199,0.38731986379829597,3.5330742906141994,1.8126852672146507,0.7246197983929721,5.855012980845159,8.136508447439535,0])

class CamVidDataset(Dataset):
    _CLASSES=['sky', 'building', 'pole', 'road', 'pavement',
           'tree', 'signsymbol', 'fence', 'car',
           'pedestrian', 'bicyclist', 'unlabelled']
    def __init__(
            self,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self._CLASSES.index(cls.lower()) for cls in classes_camvid]

        # print(self.class_values)
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        ## (720, 960, 3)
        # print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ## (720, 960, 3)
        #print(image.shape)

        mask = cv2.imread(self.masks_fps[i], 0)

        ##### 调试 ####
        #mask_t=torch.from_numpy(mask)
        #print(mask_t.size())
        # torch.Size([360, 480])

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]

        mask = np.stack(masks, axis=-1).astype('float')

        ##### 调试 ####
        # mask_t=torch.from_numpy(mask)
        # print(mask_t.size())
        # torch.Size([360, 480, 12])

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, self.images_fps[i]


    def __len__(self):
        return len(self.ids)

class CamVidTransform:
    def get_training_augmentation(self):
        train_transform = [

            albu.HorizontalFlip(p=0.5),
            # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            albu.PadIfNeeded(min_height=720, min_width=960, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255),
                             mask_value=0, p=1),

            albu.OneOf(
                [
                    albu.RandomSizedCrop([240, 720], 480, 640, w2h_ratio=4 / 3, interpolation=1, p=1.0),
                    albu.RandomCrop(height=480, width=640, always_apply=True),
                    #albu.RandomSizedCrop([240, 720], 320, 320, w2h_ratio=4 / 3, interpolation=1, p=1.0),
                    #albu.RandomCrop(height=320, width=320, always_apply=True),
                ],
                p=1.0,
            ),

            albu.GaussNoise(p=0.1),
            albu.Perspective(p=0.1),
        ]

        """
                albu.OneOf(
                    [
                        albu.CLAHE(p=1),
                        #albu.RandomBrightness(p=1),
                        albu.RandomBrightnessContrast(p=1),
                        albu.RandomGamma(p=1),
                    ],
                    p=0.1,
                ),

                albu.OneOf(
                    [
                        albu.Sharpen(p=1),
                        albu.Blur(blur_limit=3, p=1),
                        albu.MotionBlur(blur_limit=3, p=1),
                    ],
                    p=0.1,
                ),

                albu.OneOf(
                    [
                        #albu.RandomContrast(p=1),
                        albu.RandomBrightnessContrast(p=1),
                        albu.HueSaturationValue(p=1),
                    ],
                    p=0.1,
                ),

            ]"""
        return albu.Compose(train_transform)

    def get_validation_augmentation(self):
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            albu.PadIfNeeded(720, 960),
            #albu.CenterCrop(height=720, width=960, always_apply=True),
            albu.CenterCrop(height=640, width=960, always_apply=True),
        ]

        return albu.Compose(test_transform)

    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def get_preprocessing(self,preprocessing_fn):
        """Construct preprocessing transform

        Args:
            preprocessing_fn (callbale): data normalization function
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose

        """

        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=self.to_tensor, mask=self.to_tensor),
        ]

        return albu.Compose(_transform)


def formatting_label_camvid(mask):
    """
    palette = {
        0:(42,23,4),
        1: (3, 14, 234),
        2:(0,0,100),
        3:(50,150,210),
        4:(1,43,0),
        5:(100,79,100),
        6:(5,98,153),
        7:(88,65,13),
        8: (255, 255, 255),
        9:(100,0,0),
        10:(41,0,24),
        11:(0,0,0),
    }

    palette={
        0: (128, 128, 128),  # sky
        1: (128, 0, 0),  # building
        2: (192, 192, 128),  # column_pole
        3: (128, 64, 128),  # road
        4: (0, 0, 192),  # sidewalk
        5: (128, 128, 0),  # Tree
        6: (192, 128, 128),  # SignSymbol
        7: (64, 64, 128),  # Fence
        8: (64, 0, 128),  # Car
        9: (64, 64, 0),  # Pedestrian
        10: (0, 128, 192),  # Bicyclist
        11: (0, 0, 0)  # Void
    }
    """

    palette = {
        0: (102, 216, 249),  #
        1: (115, 109, 60),  #
        2: (192, 192, 128),  #
        3: (125, 125, 125),  #
        4: (0, 0, 192),  #
        5: (30, 154, 12),  #
        6: (249, 219, 47),  #
        7: (64, 64, 128),  #
        8: (64, 0, 128),  #
        9: (255, 16, 16),  #
        10: (0, 128, 192),  #
        11: (236, 236, 236)  #
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


import matplotlib.pyplot as plt
if __name__ == "__main__":
    # same image with different random transforms
    import torch
    ImgTrans = CamVidTransform()
    weights=class_weight_camvid

    test_dataset = CamVidDataset(
        "E:/data/camvid/train/",
        "E:/data/camvid/trainannot/",
        #augmentation=ImgTrans.get_training_augmentation(),
        augmentation=ImgTrans.get_validation_augmentation(),
        )
    data, label, filename=test_dataset[92]

    plt.subplot(1,2, 1).axis('off')
    plt.imshow(data)

    plt.subplot(1, 2, 2).axis('off')
    plt.imshow(formatting_label_camvid(torch.from_numpy(label).transpose(1,2).transpose(0,1)))
    plt.show()







