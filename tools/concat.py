import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1枚の画像と1枚のマスクを読み込む
img = cv2.imread("daiichi/100cmオルソresize.jpg")
mask = cv2.imread("results/100cmオルソresize_denoise.jpg", cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
img[mask == 255] = [0, 0, 255]
cv2.imwrite("results/masked.jpg", img)
