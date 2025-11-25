from PIL import Image
import numpy as np

def rd_kmeans(pixels, K, lam=0.0, iters=10, seed=0):
    rng = np.random.default_rng(seed)
    N = pixels.shape[0]

    centroids = pixels[rng.choice(N, size=K, replace=False)].copy()
    labels = np.zeros(N, dtype=int)

    for _ in range(iters):
        counts = np.bincount(labels, minlength=K).astype(float)
        probs = counts / max(1.0, counts.sum())
        probs = np.maximum(probs, 1e-12)      
        rates = -np.log2(probs)                  

        x2 = np.sum(pixels * pixels, axis=1, keepdims=True)      
        c2 = np.sum(centroids * centroids, axis=1, keepdims=True).T 
        xc = pixels.dot(centroids.T)                                
        D = x2 - 2 * xc + c2                                        

        RD = D + lam * rates.reshape(1, -1)
        
        new_labels = np.argmin(RD, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for k in range(K):
            members = pixels[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)
            else:
                centroids[k] = pixels[rng.integers(0, N)]

    return centroids, labels

def quantize_image(input_path, K, lam, iters=10, out_path="D:/VS Code/MMP/rd_output.jpg"):
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img).astype(float)            
    H, W = arr.shape[:2]
    pixels = arr.reshape(-1, 3)                  

    cent, labels = rd_kmeans(pixels, K, lam, iters=iters)
    recon = cent[labels].reshape(H, W, 3).astype(np.uint8)
    Image.fromarray(recon).save(out_path)
    print(f"Saved → {out_path}")

if __name__ == "__main__":
    K = int(input("Enter the number of colors (K): "))
    lam = float(input("Enter lambda (trade-off factor): "))
    quantize_image("D:\VS Code\MMP\image1.jpg", K, lam, iters=1rd_output
