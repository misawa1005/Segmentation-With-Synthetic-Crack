import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from MakeCrack.segmentation import segmentation


def draw_smooth_final_crack(
    x: int,
    y: int,
    angle: int,
    mask: np.ndarray,
    width: int,
    height: int,
    current_thickness: int,
    min_thickness=5,
    max_thickness=10,
    step_length=None,
    min_angle=-30,
    max_angle=30,
    branch_prob=0.08,
    angle_update_prob=0.3,
    depth=0,
    max_depth=100,
):
    """
    指定されたマスク上に変動する太さと分岐を持つひびを再帰的に描画する関数。

    Args:
    - x, y: ひびの開始点の座標
    - angle: ひびの開始時の進行方向を表す角度（0〜360度）
    - mask: ひびが描かれる画像マスク
    - width, height: マスクの幅と高さ
    - current_thickness: 現在のひびの太さ
    - min_thickness: ひびの最小の太さ
    - max_thickness: ひびの最大の太さ
    - step_length: 各ステップでのひびの進行距離
    - angle_range: ひびの方向が変わる際の角度の範囲
    - branch_prob: ひびが分岐する確率
    - depth: 現在の再帰の深さ
    - max_depth: 再帰の最大の深さ
    - non_crack_mask: ひびが描かれるべきでない領域を表すマスク画像
    """
    if depth > max_depth:
        return

    if step_length is None:
        step_length = min(mask.shape[:2]) // 30

    new_angle = angle
    for _ in range(np.random.randint(30, 100)):
        dx = int(step_length * np.cos(np.radians(new_angle)))
        dy = int(step_length * np.sin(np.radians(new_angle)))

        end_x, end_y = x + dx, y + dy

        delta_thickness = np.random.uniform(-0.4, 0.2)
        current_thickness = np.clip(
            current_thickness + delta_thickness, min_thickness, max_thickness
        )

        cv2.line(
            mask, (x, y), (end_x, end_y), 0, int(round(current_thickness))
        )

        x, y = end_x, end_y

        if np.random.rand() < angle_update_prob:
          angle = angle + np.random.randint(min_angle//2, max_angle//2)
        new_angle = angle + np.random.randint(min_angle, max_angle)


        if (
            x < 0
            or x >= width
            or y < 0
            or y >= height
        ):
            break
        if current_thickness + delta_thickness < min_thickness:
            continue

        if np.random.rand() < branch_prob:
            branch_angle = angle + np.random.choice([-1, 1]) * np.random.randint(
                30, 70
            )
            branch_prob = branch_prob / 2
            draw_smooth_final_crack(
                x,
                y,
                branch_angle,
                mask,
                width,
                height,
                current_thickness,
                min_thickness,
                max_thickness,
                step_length,
                min_angle,
                max_angle,
                branch_prob,
                depth + 1,
            )


def generate_final_crack(img, non_crack_mask=None, angle_range=60, max_depth=30, step_length=None) -> np.ndarray:
    """
    指定された幅と高さの黒色画像を作成し、ひびを描画する関数

    Args:
    - img: ひびを描画する画像
    - mask: ひびを描画する領域を表すマスク画像

    Returns:
    - ひびのマスク画像
    """
    height, width = img.shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)

    x, y = get_initial(width), get_initial(height)

    initial_angle, position = get_initial_angle(x, y, width, height)
    initial_thickness = np.random.randint(1, 3)

    draw_smooth_final_crack(
        x, y, initial_angle, mask, width, height, initial_thickness
    )

    if np.random.rand() < 0.1:
        position_new = position
        while position == position_new:
            x, y = get_initial(width), get_initial(height)
            initial_angle, position_new = get_initial_angle(
                x, y, width, height
            )

        initial_thickness = np.random.randint(1, 3)
        draw_smooth_final_crack(
            x,
            y,
            initial_angle,
            mask,
            width,
            height,
            initial_thickness,
            step_length=step_length,
            max_depth=max_depth,
            min_angle=-angle_range//2,
            max_angle=angle_range//2,
            branch_prob=0,
        )
    if non_crack_mask is not None:
        mask[non_crack_mask == 0] = 255
    crack_img = project_mask(img, mask)
    mask_img_reverse = cv2.bitwise_not(mask)

    return crack_img, mask_img_reverse


def project_mask(
    input_img: np.ndarray,
    mask_img: np.ndarray,
    kernel_size: int = 7,
    probability: float = 0.1,
):
    """
    input_imageにmask_imgを投影する関数
    Args:
    - input_img: 投影先の画像
    - mask_img: 投影する画像
    - kernel_size: 平滑化のカーネルサイズ
    Returns:
    - 投影後の画像
    """
    mask_img_3ch = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)

    crack_image = input_img.copy()
    crack_image[mask_img_3ch == 0] = mask_img_3ch[mask_img_3ch == 0]

    white_pixels = np.where(mask_img == 0)
    random_values = np.random.rand(len(white_pixels[0]))
    crack_image[
        white_pixels[0][random_values < probability],
        white_pixels[1][random_values < probability],
    ] = (
        np.mean(crack_image, axis=(0, 1)) + np.random.randint(0, 100) + 255
    ) / 3

    blur_img = cv2.blur(crack_image, (kernel_size, kernel_size))
    output_image = input_img.copy()
    output_image[mask_img_3ch == 0] = blur_img[mask_img_3ch == 0]
    return output_image


def get_initial(length: int) -> int:
    """
    画像内の周囲5%ないから初期地点を求める
    Args:
    - length: サイズ
    Returns:
    - random_number: 乱数
    """
    n = length // 20
    ranges = [(-n, n), (length - n, length+n)]
    random_numbers_from_ranges = [
        np.random.randint(start, end + 1) for start, end in ranges
    ]
    random_number = np.random.choice(random_numbers_from_ranges)
    random_number = np.clip(random_number, 0, length - 1)
    return random_number


def get_initial_angle(x, y, width, height):
    x_dist = x - width / 2
    y_dist = y - height / 2
    if x_dist >= 0 and y_dist < 0:
        angle = np.random.randint(100, 170)
        position = "top_right"
    elif x_dist < 0 and y_dist < 0:
        angle = np.random.randint(10, 80)
        position = "top_left"
    elif x_dist < 0 and y_dist >= 0:
        angle = np.random.randint(280, 350)
        position = "bottom_left"
    else:
        angle = np.random.randint(190, 260)
        position = "bottom_right"
    return angle, position


def generate_crack(img_path):
    img = cv2.imread(str(img_path))
    #mask = segmentation(str(img_path), "wall")
    crack_img, mask_img = generate_final_crack(img)
    cv2.imwrite(f"data/makecrack/make/imgs/{img_path.name}", crack_img)
    cv2.imwrite(f"data/makecrack/make/masks/{img_path.stem}.png", mask_img)


def multiprocess_time(img_dir):
    img_paths = list(img_dir.glob("*.jpg"))
    num_processes = min(cpu_count(), len(img_paths))

    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(generate_crack, img_paths), total=len(img_paths)))


def normal_time(img_dir):
    img_paths = img_dir.glob("*.jpg")

    for img_path in tqdm(img_paths):
        generate_crack(img_path)


if __name__ == "__main__":
    img_dir = Path("data/makecrack/normal/")
    multiprocess_time(img_dir)
