import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.stats import pearsonr

# Load the two images (grayscale)
image1 = io.imread('bluec.tif', as_gray=True)
image2 = io.imread('greenc.tif', as_gray=True)

# Check if the images are of the same size, if not resize
if image1.shape != image2.shape:
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# Calculate Pearson's Correlation Coefficient for each pixel (sliding window approach)
def calculate_pcc(image1, image2, window_size=3):
    half_win = window_size // 2
    pcc_map = np.zeros_like(image1)
    
    for i in range(half_win, image1.shape[0] - half_win):
        for j in range(half_win, image1.shape[1] - half_win):
            # Extract local windows around the current pixel
            window1 = image1[i-half_win:i+half_win+1, j-half_win:j+half_win+1].flatten()
            window2 = image2[i-half_win:i+half_win+1, j-half_win:j+half_win+1].flatten()
            
            # Ensure that window1 and window2 have the same length
            if len(window1) == len(window2):
                # Calculate PCC for the local window
                pcc_value, _ = pearsonr(window1, window2)
                pcc_map[i, j] = pcc_value
    
    return pcc_map

# Calculate the PCC map using a window size of 3x3
pcc_map = calculate_pcc(image1, image2, window_size=3)

# Threshold the PCC map to retain areas with PCC >= 0.93
threshold = 0.42
mask = pcc_map >= threshold

# Apply the mask to image2 to keep only colocalized areas
image2_filtered = np.zeros_like(image2)
image2_filtered[mask] = image2[mask]

# Show the results
plt.figure(figsize=(10, 5))

# Original image 2 (EdU-Alexa Fluor 488)
plt.subplot(1, 3, 1)
plt.imshow(image2, cmap='gray')
plt.title("Original Image 2 (EdU-Alexa Fluor 488)")
plt.axis('off')

# PCC Map
plt.subplot(1, 3, 2)
plt.imshow(pcc_map, cmap='hot')
plt.title("PCC Map")
plt.colorbar()
plt.axis('off')

# Filtered image 2 based on PCC >= 0.93
plt.subplot(1, 3, 3)
plt.imshow(image2_filtered, cmap='gray')
plt.title("Filtered Image 2 (PCC >= 0.93)")
plt.axis('off')

plt.tight_layout()
plt.show()

# Ensure the image is properly scaled and saved as an 8-bit TIFF
image2_filtered_8bit = cv2.normalize(image2_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
io.imsave('filtered_image2_pcc_093.tiff', image2_filtered_8bit)
