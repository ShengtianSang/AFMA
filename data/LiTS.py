import os
import cv2
from torch.utils.data import Dataset
import numpy as np
import torch
import albumentations as albu

classes_LiTS = ['liver', 'liver tumor', 'unlabeled']
class LiTS_Dataset(Dataset):
    CLASSES = ['background', 'liver', 'liver tumor']

    def __init__(
            self,
            images_dir,
            masks_dir,
            train_or_val=True,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.train_or_val=train_or_val

        if self.train_or_val == True:
            self.ids = os.listdir(images_dir)
            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids if "volume" in image_id]
            self.masks_fps = [os.path.join(masks_dir, image_id.replace("volume","segmentation")) for image_id in self.images_fps]
        else:
            self.ids = os.listdir(images_dir)
            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids if "volume" in image_id]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        if self.train_or_val == True:
            image = np.load(self.images_fps[i])
            image= self.normalise(image)
            mask = np.load(self.masks_fps[i])
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')

            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            image=self.dimension_swith(torch.from_numpy(image))
            mask=self.dimension_swith(torch.from_numpy(mask)).long()
            return image, mask

        else:
            image = np.load(self.images_fps[i])
            image = self.normalise(image)
            image_file_name=self.images_fps[i]

            image = self.dimension_swith(torch.from_numpy(image))

            return image,image_file_name

    def __len__(self):
        return len(self.images_fps)

    def normalise(self, image):
        # normalise and clip images -250 to 250
        np_img = image
        np_img = np.clip(np_img, -250, 250.).astype(np.float32)

        return np_img

    def dimension_swith(self,x):
        return torch.transpose(torch.transpose(x,0,2),1,2)


class LiTSTransform:

    def get_training_augmentation(self):
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
            albu.RandomCrop(height=512, width=512, always_apply=True),

            albu.GaussNoise(p=0.2),
            albu.Perspective(p=0.5),

            albu.OneOf(
                [
                    #albu.CLAHE(p=1),
                    #albu.RandomBrightness(p=1),
                    albu.RandomBrightnessContrast(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.Sharpen(p=1),
                    albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    #albu.RandomContrast(p=1),
                    albu.RandomBrightnessContrast(p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
        return albu.Compose(train_transform)


    def get_validation_augmentation(self):
        test_transform = [
            albu.PadIfNeeded(512, 512),
            albu.CenterCrop(height=512, width=512, always_apply=True),
        ]
        return albu.Compose(test_transform)

    def get_preprocessing(self,preprocessing_fn):
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            #albu.Lambda(image=self.to_tensor, mask=self.to_tensor),
        ]
        return albu.Compose(_transform)
