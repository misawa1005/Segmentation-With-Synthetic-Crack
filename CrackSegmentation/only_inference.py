import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from dataloader import CustomSegmentationDataset
from segmentation import CrackModel
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    iou: float


def calculate_iou(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    予測結果と正解ラベルからIoUを計算する関数
    Args:
        y_pred (np.ndarray): 予測結果
        y_true (np.ndarray): 正解ラベル
    Returns:
        iou_score (float): IoU
    """
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def calculate_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Metrics:
    """
    予測結果と正解ラベルから各種指標を計算する関数
    Args:
        y_pred (np.ndarray): 予測結果
        y_true (np.ndarray): 正解ラベル
    Returns:
        metrics (Metrics): 各種指標
    """
    y_pred = np.where(y_pred > 0, 1, 0)
    y_true = np.where(y_true > 0, 1, 0)
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    true_positives = np.sum(y_pred_flat * y_true_flat)
    true_negatives = np.sum((1 - y_pred_flat) * (1 - y_true_flat))
    false_positives = np.sum(y_pred_flat * (1 - y_true_flat))
    false_negatives = np.sum((1 - y_pred_flat) * y_true_flat)

    accuracy = (true_positives + true_negatives) / len(y_pred_flat)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )
    iou = calculate_iou(y_pred, y_true)

    return Metrics(accuracy, precision, recall, f1, iou)


def inference(
    checkpoint_path: Path,
    data_dir: Path,
    batch_size: int,
    output_dir: Path = Path("results"),
    mode: str = "infer",
):
    """
    推論を行う関数
    Args:
        checkpoint_path (Path): チェックポイントのパス
        data_dir (Path): データセットのパス
        batch_size (int): バッチサイズ
        output_dir (Path): 推論結果の出力先
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    model = CrackModel.load_from_checkpoint(
        str(checkpoint_path),
        arch="FPN",
        encoder_name="resnet34",
        in_channels=3,
        out_classes=1,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((448, 448)),
        ]
    )

    test_dataset = CustomSegmentationDataset(
        data_dir=str(data_dir), mode=mode, transform=transform
    )
    n_cpu = os.cpu_count()
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu
    )

    model.eval()
    image_idx = 0
    if test_dataset.mode == "infer":
        with torch.no_grad():
            for batch in test_dataloader:
                logits = model(batch)
                pr_masks = logits.sigmoid()

                for i, (pr_mask, img_name) in enumerate(
                    zip(
                        pr_masks,
                        test_dataset.imgs[image_idx:],
                    )
                ):
                    mask_normalized = cv2.normalize(
                        pr_mask.numpy().squeeze(), None, 0, 255, cv2.NORM_MINMAX
                    )
                    mask_uint8 = mask_normalized.astype(np.uint8)
                    cv2.imwrite(f'{str(output_dir)}/{img_name.split(".")[0]}.png', mask_uint8)
                image_idx += len(batch)
    else:
        scores = []
        with torch.no_grad():
            for batch in test_dataloader:
                logits = model(batch["image"])
                pr_masks = logits.sigmoid()

                for i, (gr_mask, pr_mask, image) in enumerate(
                    zip(
                        batch["mask"],
                        pr_masks,
                        batch["image"],
                    )
                ):
                    mask_normalized = cv2.normalize(
                        pr_mask.numpy().squeeze(), None, 0, 255, cv2.NORM_MINMAX
                    )
                    mask_uint8 = mask_normalized.astype(np.uint8)
                    scores.append(
                        calculate_metrics(
                            mask_uint8, gr_mask.numpy().squeeze().astype(np.uint8)
                        )
                    )
                    if (i+image_idx)%30 == 0:
                        plt.figure(figsize=(10, 5))
                        plt.subplot(1, 3, 1)
                        plt.imshow(image.numpy().transpose(1, 2, 0))
                        plt.title("Image")
                        plt.axis("off")

                        plt.subplot(1, 3, 2)
                        plt.imshow(gr_mask.numpy().squeeze())
                        plt.title("Ground truth")
                        plt.axis("off")

                        plt.subplot(1, 3, 3)
                        plt.imshow(mask_uint8)
                        plt.title("Prediction")
                        plt.axis("off")
                        plt.savefig(f"{str(output_dir)}/inference_{i+image_idx}.png")
                        plt.close()
                image_idx += len(batch)
        print(f"Accuracy: {np.mean([score.accuracy for score in scores])}")
        print(f"Precision: {np.mean([score.precision for score in scores])}")
        print(f"Recall: {np.mean([score.recall for score in scores])}")
        print(f"F1 score: {np.mean([score.f1_score for score in scores])}")
        print(f"mIoU: {np.mean([score.iou for score in scores])}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--mode", type=str, default="infer")
    args = parser.parse_args()

    inference(
        Path(args.checkpoint_path),
        Path(args.data_dir),
        args.batch_size,
        Path(args.output_dir),
        args.mode,
    )
