from segmentation import CrackModel
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataloader import CustomSegmentationDataset
import os
import torch
from torchvision import transforms


if __name__ == "__main__":
    model = CrackModel("FPN", "resnet34", in_channels=3, out_classes=1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),  # サイズは必要に応じて変更してください
    ])

    train_dataset = CustomSegmentationDataset(data_dir='data/train', transform=transform)
    valid_dataset = CustomSegmentationDataset(data_dir='data/train', transform=transform)
    test_dataset = CustomSegmentationDataset(data_dir='data/test', transform=transform)

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)


    trainer = pl.Trainer(
        max_epochs=5,
    )

    trainer.fit(
        model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=valid_dataloader,
    )