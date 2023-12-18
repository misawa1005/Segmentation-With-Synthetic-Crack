from segmentation import CrackModel
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataloader import CustomSegmentationDataset
import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np

if __name__ == "__main__":
    checkpoint_path = 'lightning_logs/version_8/checkpoints/epoch=4-step=3004.ckpt'

    model = CrackModel.load_from_checkpoint(checkpoint_path, arch='FPN', encoder_name='resnet34', in_channels=3, out_classes=1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((448, 448)),
    ])

    test_dataset = CustomSegmentationDataset(data_dir="output", mode="infer", transform=transform)
    n_cpu = os.cpu_count()
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

    model.eval()
    image_idx = 0  # 保存する画像のインデックス
    with torch.no_grad():
        for batch in test_dataloader:

            logits = model(batch)
            pr_masks = logits.sigmoid()

            for i, (image, pr_mask, img_name) in enumerate(zip(batch, pr_masks, test_dataset.imgs[image_idx:])):
                mask_normalized = cv2.normalize(pr_mask.numpy().squeeze(), None, 0, 255, cv2.NORM_MINMAX)
                mask_uint8 = mask_normalized.astype(np.uint8)
                cv2.imwrite(f'results/{img_name.split(".")[0]}.png', mask_uint8)
            image_idx += len(batch)
