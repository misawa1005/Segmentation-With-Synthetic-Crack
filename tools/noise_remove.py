import cv2
import numpy as np
from pathlib import Path
from typing import List
from MakeCrack.segmentation import segmentation


def noise_remove(img_path: Path, mask_path: Path, save_dir: Path) -> None:
    """
    画像の直線ノイズを除去する
    Args:
        img_path (Path): 画像のパス
        mask_path (Path): マスクのパス
    """
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    min_shape = min(img.shape[:2])
    fx = 4000 / min_shape
    resize_img = cv2.resize(img, None, fx=fx, fy=fx)
    line_mask = _detect_line_mask(resize_img)
    seg_mask = segmentation(img, "wall", resize_img)
    seg_mask = cv2.bitwise_not(seg_mask)
    line_mask = cv2.bitwise_or(line_mask, seg_mask)
    line_mask = cv2.resize(line_mask, (mask.shape[1], mask.shape[0]))
    mask[line_mask == 255] = 0
    cv2.imwrite(f"{save_dir/img_path.stem}_denoise.jpg", mask)


def _detect_line_mask(
    img: np.ndarray, minLineLength: int = 80, maxLineGap: int = 10
) -> np.ndarray:
    """
    Cany法で直線を検出し、マスクを作成する
    Args:
        img (np.ndarray): 画像
        minLineLength (int): 最小の線分の長さ
        maxLineGap (int): 線分を結合する際の最大の距離
    Returns:
        np.ndarray: マスク画像
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=minLineLength,
        maxLineGap=maxLineGap,
    )
    lines = _merge_lines(lines, angle_threshold=10, distance_threshold=100)
    line_img = np.zeros_like(img)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 4)
    line_mask = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    line_mask = cv2.morphologyEx(
        line_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), line_mask, iterations=5
    )
    line_mask = cv2.dilate(line_mask, np.ones((5, 5), np.uint8), iterations=5)
    return line_mask


def _merge_lines(
    lines: np.ndarray, angle_threshold: int = 10, distance_threshold: int = 20
) -> np.ndarray:
    """
    線分を統合する
    Args:
        lines (list): 線分のリスト
        angle_threshold (int): 統合する角度の閾値
        distance_threshold (int): 統合する距離の閾値
    Returns:
        list: 統合された線分のリスト
    """
    if lines is None:
        return []

    # 線分を角度で分類
    lines_by_angle = {}
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            rounded_angle = int(round(angle / angle_threshold) * angle_threshold)
            if rounded_angle not in lines_by_angle:
                lines_by_angle[rounded_angle] = []
            lines_by_angle[rounded_angle].append((x1, y1, x2, y2))

    merged_lines = []
    for angle, lines in lines_by_angle.items():
        # 同じ角度の線分を距離で統合
        while len(lines) > 0:
            line = lines.pop()
            x1, y1, x2, y2 = line
            for i, (x3, y3, x4, y4) in enumerate(lines):
                if (
                    np.linalg.norm(np.array([x1, y1]) - np.array([x3, y3]))
                    < distance_threshold
                    or np.linalg.norm(np.array([x1, y1]) - np.array([x4, y4]))
                    < distance_threshold
                    or np.linalg.norm(np.array([x2, y2]) - np.array([x3, y3]))
                    < distance_threshold
                    or np.linalg.norm(np.array([x2, y2]) - np.array([x4, y4]))
                    < distance_threshold
                ):
                    x1, y1, x2, y2 = min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4)
                    lines.pop(i)
                    break
            merged_lines.append([x1, y1, x2, y2])

    return np.array([merged_lines])


if __name__ == "__main__":
    noise_remove(Path("daiichi/100cmオルソresize.jpg"), Path("daiichi/mask.png"), Path("results"))
