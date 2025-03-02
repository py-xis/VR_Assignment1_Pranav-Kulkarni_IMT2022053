import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

INPUT_DIR = '../input/'
OUTPUT_DIR = '../results/'

def readImage(imageName: str) -> np.ndarray:
    """
    Reads an image from disk
    :param imageName: Name of the image
    :return: Image as a numpy array
    """
    try:
        image = cv2.imread(INPUT_DIR + imageName, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Image '{imageName}' not found in {INPUT_DIR}.")
        return image
    except Exception as e:
        raise RuntimeError(f"Error reading image '{imageName}': {e}")

def writeImage(imageName: str, image: np.ndarray) -> bool:
    """
    Writes an image to disk
    :param imageName: Name of the image
    :param image: Image to write
    :return: True if successful, False otherwise
    """
    try:
        success = cv2.imwrite(OUTPUT_DIR + imageName, image)
        if not success:
            raise IOError(f"Failed to write image '{imageName}' to {OUTPUT_DIR}.")
        return True
    except Exception as e:
        raise RuntimeError(f"Error writing image '{imageName}': {e}")

def plotHistogram(image: np.ndarray, title: str) -> None:
    """
    Plots the histogram of an image
    :param image: Input image
    :param title: Title of the plot
    :return None
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plt.figure(figsize=(6, 4))
        plt.hist(gray.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
        plt.title(title)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.grid()
        plt.savefig(OUTPUT_DIR + f"{title.replace(' ', '_').lower()}.png")
        plt.close()
    except Exception as e:
        raise RuntimeError(f"Error plotting histogram: {e}")

def applyManualThresholding(image: np.ndarray, threshold_value: int) -> np.ndarray:
    """
    Applies manual thresholding
    :param image: Input image
    :param threshold_value: Threshold value
    :return: Binary image
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        return binary
    except Exception as e:
        raise RuntimeError(f"Error applying manual thresholding: {e}")

def applyAdaptiveThresholding(image: np.ndarray) -> np.ndarray:
    """
    Applies adaptive thresholding
    :param image: Input image
    :return: Binary image
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        return binary
    except Exception as e:
        raise RuntimeError(f"Error applying adaptive thresholding: {e}")

def applyOtsuThresholding(image: np.ndarray) -> (np.ndarray, float):
    """
    Applies Otsu's thresholding
    Returns the binary image and the threshold value
    :param image: Input image
    :return: Binary image and threshold value
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        otsu_threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary, otsu_threshold
    except Exception as e:
        raise RuntimeError(f"Error applying Otsu thresholding: {e}")

def watershedSegmentation(image: np.ndarray, clipLimit: float, tileGrid: int, blurKernel: int, morphIterations: int, dilateIterations: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """Performs watershed segmentation"""
    try:
        if image is None:
            raise ValueError("Error: Image is None")

        image = image.astype(np.float32) / 255.0
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGrid, tileGrid))
        gray = clahe.apply((gray * 255).astype(np.uint8))

        smoothened = cv2.GaussianBlur(gray, (blurKernel, blurKernel), 0)
        _, thresholded = cv2.threshold(smoothened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=morphIterations)

        background = cv2.dilate(opening, kernel, iterations=dilateIterations)
        distTransform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, foreground = cv2.threshold(distTransform, 0.1 * distTransform.max(), 255, 0)
        foreground = np.uint8(foreground)

        unknown = cv2.subtract(background, foreground)
        _, markers = cv2.connectedComponents(foreground)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = np.int32(markers)
        cv2.watershed((image * 255).astype(np.uint8), markers)

        segmentedImage = np.zeros_like(image)
        colors = {marker: np.random.randint(0, 255, 3).tolist() for marker in range(2, markers.max() + 1)}
        for marker in range(2, markers.max() + 1):
            segmentedImage[markers == marker] = colors[marker]

        imageWithBoundaries = (image * 255).astype(np.uint8)
        imageWithBoundaries[markers == -1] = [255, 255, 255]

        return segmentedImage, imageWithBoundaries, markers
    except Exception as e:
        raise RuntimeError(f"Error in watershed segmentation: {e}")

def getCoins(markers: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Extracts individual coins from the segmented image"""
    try:
        coins = []
        for mark in range(2, markers.max() + 1):
            mask = np.uint8(markers == mark) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                coin = image[y:y+h, x:x+w]
                coins.append(cv2.cvtColor(coin, cv2.COLOR_BGR2RGB))
        return coins
    except Exception as e:
        raise RuntimeError(f"Error extracting coins: {e}")

if __name__ == "__main__":
    print("Starting...")

    try:
        image = readImage('image.jpg')

        if image.shape[-1] == 4:  
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Create directories
        segmentation_dir = os.path.join(OUTPUT_DIR, 'segmentation/')
        coins_dir = os.path.join(segmentation_dir, 'coins/')
        os.makedirs(segmentation_dir, exist_ok=True)
        os.makedirs(coins_dir, exist_ok=True)

        # Histogram plotting
        plotHistogram(image, "Original Image Histogram")

        # Apply different thresholding techniques
        manualThresholded = applyManualThresholding(image, 250)
        adaptiveThresholded = applyAdaptiveThresholding(image)
        otsuThresholded, otsuValue = applyOtsuThresholding(image)
        segmented, boundaries, markers = watershedSegmentation(image, 2.50, 6, 55, 63, 147)
        coins = getCoins(markers, image) 

        print(f"\033[94mOtsu's Threshold Value: {otsuValue}\033[0m")

        # Save thresholded images
        writeImage(os.path.join(segmentation_dir, 'manual_threshold.png'), manualThresholded)
        writeImage(os.path.join(segmentation_dir, 'adaptive_threshold.png'), adaptiveThresholded)
        writeImage(os.path.join(segmentation_dir, 'otsu_threshold.png'), otsuThresholded)
        writeImage(os.path.join(segmentation_dir, 'segmented_image.png'), segmented)

        # Save extracted coins
        for i, coin in enumerate(coins):
            writeImage(os.path.join(coins_dir, f'coin_{i + 1}.jpg'), coin)
            print(f"\033[94mCoin saved: file://{coins_dir}/coin_{i + 1}.jpg\033[0m")

        print("\033[92mUse Ctrl/Cmd + Click on the links to open images in a new tab\033[0m")

    except Exception as e:
        print(f"\033[91mError: {e}\033[0m")

    print("Done!")