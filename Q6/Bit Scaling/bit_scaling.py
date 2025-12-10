import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("D:/VS Code/MMP/Q6/input2.jpg").convert("L")
original = np.array(img)

reconstructed = original & 7
reconstructed_scaled = (reconstructed * 255) // 7

difference = np.abs(original - reconstructed_scaled)

plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(original, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("LSB Reconstruction")
plt.imshow(reconstructed_scaled, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Difference Image")
plt.imshow(difference, cmap="gray")
plt.axis("off")

plt.show()

# 6. Save output images
Image.fromarray(reconstructed_scaled).save("D:/VS Code/MMP/Q6/lsb_reconstructed2.png")
Image.fromarray(difference).save("D:/VS Code/MMP/Q6/difference2.png")
