from segmentation import CrackModel
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataloader import CustomSegmentationDataset
import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


if __name__ == "__main__":
    checkpoint_path = 'lightning_logs/version_6/checkpoints/epoch=4-step=3004.ckpt'

    model = CrackModel.load_from_checkpoint(checkpoint_path, arch='FPN', encoder_name='resnet34', in_channels=3, out_classes=1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),  # サイズは必要に応じて変更してください
    ])

    test_dataset = CustomSegmentationDataset(data_dir='data/test', transform=transform)
    n_cpu = os.cpu_count()
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

    batch = next(iter(test_dataloader))
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_masks = logits.sigmoid()

    for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
        plt.title("Prediction")
        plt.axis("off")

        plt.show()