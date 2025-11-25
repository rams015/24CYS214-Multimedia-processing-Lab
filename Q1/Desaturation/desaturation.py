import imageio 
import numpy as np

img = imageio.imread("D:\VS Code\MMP\image.jpg")

gray = (img[...,0] * 0.299 + img[...,1] * 0.587 + img[...,2] * 0.114)

gray = gray.astype(np.uint8)

imageio.imwrite("D:\VS Code\MMP\greyscale.jpg",gray)
