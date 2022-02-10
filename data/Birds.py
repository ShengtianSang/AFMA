import os
import cv2
from torch.utils.data import Dataset
import numpy as np
import albumentations as albu

classes_bird = ['bird','unlabeled']

class Bird_Dataset(Dataset):
    #CLASSES = ['bird', 'unlabeled']
    CLASSES = ['bird']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.masks_fps = []
        self.images_fps = []
        masks_files = os.listdir(masks_dir)
        for file in masks_files:
            each_mask_dir = os.path.join(masks_dir, file)
            each_img_dir = os.path.join(images_dir, file)
            for img in os.listdir(each_mask_dir):
                self.masks_fps.append(os.path.join(each_mask_dir, img))
                self.images_fps.append(os.path.join(each_img_dir, img.split(".png")[0]+".jpg"))

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        # print(self.class_values)
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]

        mask = np.stack(masks, axis=-1).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images_fps)


class BirdTransform:
    def get_training_augmentation(self):
        train_transform = [

            albu.HorizontalFlip(p=0.5),

            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
            albu.RandomCrop(height=320, width=320, always_apply=True),

            albu.IAAAdditiveGaussianNoise(p=0.2),
            albu.IAAPerspective(p=0.5),

            albu.OneOf(
                [
                    albu.CLAHE(p=1),
                    albu.RandomBrightness(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.IAASharpen(p=1),
                    albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.RandomContrast(p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
        return albu.Compose(train_transform)


    def get_validation_augmentation(self):
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            albu.PadIfNeeded(320, 320),
            albu.CenterCrop(height=320, width=320, always_apply=True),
        ]
        return albu.Compose(test_transform)


    def to_tensor(self,x, **kwargs):
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

