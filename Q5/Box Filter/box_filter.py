import cv2
import numpy as np
import math

path = r"D:/VS Code/MMP/Q5/Torgya - Arunachal Festival.jpg"

img = cv2.imread(path)
img = img.astype(np.float32) 

box5_avg = cv2.boxFilter(img, -1, (5,5), normalize=True) # 5x5 box blur (average)

box5_sum = cv2.boxFilter(img, -1, (5,5), normalize=False) # 5x5 box blur (sum)

box20_avg = cv2.boxFilter(img, -1, (20,20), normalize=True) # 20x20 box blur (average)

box20_sum = cv2.boxFilter(img, -1, (20,20), normalize=False) # 20x20 box blur (sum)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sigma = np.std(gray)
print("Sigma =", sigma)

size = int(round(2 * math.pi * sigma))

if size < 3:
    size = 3
if size % 2 == 0:
    size += 1

print("Gaussian kernel size =", size)

radius = size // 2
x = np.arange(-radius, radius + 1, 1, dtype=np.float32)
gauss_unnorm = np.exp(-(x**2) / (2 * sigma * sigma + 1e-6))# Gaussian formula

gauss_norm = gauss_unnorm / np.sum(gauss_unnorm) # Normalized Gaussian (sum = 1)

print("Gaussian kernel (normalized) sum =", np.sum(gauss_norm))
gauss_sep_unnorm = cv2.sepFilter2D(img, -1, gauss_unnorm, gauss_unnorm) #Unnormalized Gaussian blur

gauss_sep_norm = cv2.sepFilter2D(img, -1, gauss_norm, gauss_norm) # Normalized Gaussian blur

cv2.imwrite("D:/VS Code/MMP/Q5/box5_avg.jpg", box5_avg)
cv2.imwrite("D:/VS Code/MMP/Q5/box5_sum.jpg", box5_sum)
cv2.imwrite("D:/VS Code/MMP/Q5/box20_avg.jpg", box20_avg)
cv2.imwrite("D:/VS Code/MMP/Q5/box20_sum.jpg", box20_sum)
cv2.imwrite("D:/VS Code/MMP/Q5/gaussian_unnormalized.jpg", gauss_sep_unnorm)
cv2.imwrite("D:/VS Code/MMP/Q5/gaussian_normalized.jpg", gauss_sep_norm)

print("All images saved!")
