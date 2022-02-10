import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as albu

classes_camvid = ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled']

class_weight_camvid=torch.tensor([0.25652730831033116,0.18528266685745448,4.396287575365375,0.1368693220338383,0.9184731310542199,0.38731986379829597,3.5330742906141994,1.8126852672146507,0.7246197983929721,5.855012980845159,8.136508447439535,1.0974099206087582])

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

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[i], 0)

        masks = [(mask == v) for v in self.class_values]

        mask = np.stack(masks, axis=-1).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

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
            albu.PadIfNeeded(min_height=720, min_width=960, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255),
                             mask_value=0, p=1),

            albu.OneOf(
                [
                    albu.RandomSizedCrop([240, 720], 480, 640, w2h_ratio=4 / 3, interpolation=1, p=1.0),
                    albu.RandomCrop(height=480, width=640, always_apply=True),
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
            albu.PadIfNeeded(720, 960),
            albu.CenterCrop(height=640, width=960, always_apply=True),
        ]

        return albu.Compose(test_transform)

    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def get_preprocessing(self,preprocessing_fn):
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=self.to_tensor, mask=self.to_tensor),
        ]

        return albu.Compose(_transform)









