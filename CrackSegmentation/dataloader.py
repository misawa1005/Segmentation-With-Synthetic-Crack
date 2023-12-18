import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset

class CustomSegmentationDataset(Dataset):
    def __init__(self, data_dir, trimap_width=3, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.trimap_width = trimap_width
        self.mode = mode

        self.imgs = list(sorted(os.listdir(os.path.join(data_dir, "imgs"))))

        if mode != 'infer':
            self.masks = list(sorted(os.listdir(os.path.join(data_dir, "masks"))))
        else:
            self.masks = None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, "imgs", self.imgs[idx])
        image = Image.open(img_path).convert("RGB")

        if self.mode == 'infer':
            if self.transform is not None:
                image = self.transform(image)
            return image

        mask_path = os.path.join(self.data_dir, "masks", self.masks[idx])
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        mask_np = np.array(mask)
        trimap = self.generate_trimap(mask_np)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
            trimap = self.transform(Image.fromarray(trimap))

        return {
            "image": image, 
            "mask": mask, 
            "trimap": trimap
        }

    def generate_trimap(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.trimap_width, self.trimap_width))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        trimap = np.where(dilated != eroded, 127, mask)  # 127 for unknown regions
        return trimap
