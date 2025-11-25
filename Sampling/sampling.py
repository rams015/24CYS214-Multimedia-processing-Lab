from PIL import Image
import numpy as np

path = "img.jpg"

img_gray = Image.open(path).convert("L")
img_rgb  = Image.open(path).convert("RGB")

gray = np.array(img_gray)
rgb  = np.array(img_rgb)

print("\nGrayscale Resolution :", gray.shape)
print("RGB Resolution       :", rgb.shape)

def freq_sample(image, factor):
    F = np.fft.fftshift(np.fft.fft2(image))
    H, W = F.shape

    h = max(1, H // factor)
    w = max(1, W // factor)

    F_low = np.zeros_like(F)

    hs = (H // 2) - (h // 2)
    he = hs + h
    ws = (W // 2) - (w // 2)
    we = ws + w

    F_low[hs:he, ws:we] = F[hs:he, ws:we]

    out = np.abs(np.fft.ifft2(np.fft.ifftshift(F_low)))
    mn, mx = out.min(), out.max()

    if mx - mn < 1e-9:
        return out.astype(np.uint8)

    norm = (out - mn) / (mx - mn)
    return (norm * 255).astype(np.uint8)

def spatial_sample(image, factor):
    return image[::factor, ::factor].copy()

factors = [2, 4, 8, 16]

for f in factors:
    freq_out = freq_sample(gray, f)
    freq_name = f"freq_1_{f}.png"
    Image.fromarray(freq_out).save(freq_name)

    print(f"\nFrequency Sampling 1/{f}")
    print(" → Output Resolution:", freq_out.shape)
    print("Saved:", freq_name)

    spatial_out = spatial_sample(rgb, f)
    spatial_name = f"spatial_1_{f}.png"
    Image.fromarray(spatial_out).save(spatial_name)

    print(f"Spatial Sampling 1/{f}")
    print(" → Output Resolution:", spatial_out.shape)
    print("Saved:", spatial_name)

print("\nAll images saved in the current directory.")
