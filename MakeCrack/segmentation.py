import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
# segment anything
from segment_anything import SamPredictor, sam_model_registry

from GroundedSAM.segmentation import (get_grounding_output, load_image,
                                      load_image_cv2, load_model)


def segmentation(image_path:Path, text_prompt: str, img:np.ndarray = None) -> None:
    config_file = "GroundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "weights/groundingdino_swint_ogc.pth"
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    # load image
    if img is None:
        image_pil, image = load_image(image_path)
    else:
        image_pil, image = load_image_cv2(img)
    # load model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = load_model(config_file, grounded_checkpoint, device=device)

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, 0.3, 0.25, device=device
    )

    predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to("mps"))
    if img is None:
        image = cv2.imread(image_path)
    else:
        image = img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to("mps")

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to("mps"),
        multimask_output = False,
    )

    for mask in masks:
        all_mask = get_mask(mask.cpu().numpy())

    return all_mask


def get_mask(masks: [np.ndarray]) -> np.ndarray:
    """
    Args:
    - masks: segmentation masks
    Returns:
    - mask: segmentation mask
    """
    mask = np.zeros((masks.shape[1], masks.shape[2]))
    for i in range(masks.shape[0]):
        mask += masks[i]
    mask = np.where(mask > 0, 255, 0)
    return mask.astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--text_prompt", type=str, required=True)
    args = parser.parse_args()

    mask = segmentation(args.image_path, args.text_prompt)
    cv2.imwrite("mask.png", mask)