import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

INPUT_DIR = '../input/'
OUTPUT_DIR = '../results/'

def readImage(imageName: str) -> np.ndarray:
    """
    Reads an image from the disk
    :param imageName: Name of the image file
    :return: Image as a numpy array
    """
    try:
        image = cv2.imread(INPUT_DIR + imageName)
        if image is None:
            raise FileNotFoundError(f"Image '{imageName}' not found in {INPUT_DIR}.")
        return image
    except Exception as e:
        raise RuntimeError(f"Error reading image '{imageName}': {e}")

def writeImage(imageName: str, image: np.ndarray) -> bool:
    """
    Writes an image to the disk
    :param imageName: Name of the image file
    :param image: Image as a numpy array
    :return: True if the image was written successfully, False otherwise
    """
    try:
        success = cv2.imwrite(OUTPUT_DIR + imageName, image)
        if not success:
            raise IOError(f"Failed to write image '{imageName}' to {OUTPUT_DIR}.")
        return True
    except Exception as e:
        raise RuntimeError(f"Error writing image '{imageName}': {e}")

def showImage(image: np.ndarray, title: str = 'Image') -> None:
    """
    Shows an image
    :param image: Image as a numpy array
    :param title: Title of the window
    :return: None
    """
    try:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        raise RuntimeError(f"Error displaying image '{title}': {e}")

def smoothen(image: np.ndarray, kernelSize: int = 5, sigma: int = 0) -> np.ndarray:
    """
    Smoothen an image using Gaussian Blur
    :param image: Image as a numpy array
    :param kernelSize: Kernel size for the Gaussian Kernel
    :return: Smoothened image as a numpy array
    """
    try:
        return cv2.GaussianBlur(image, (kernelSize, kernelSize), 0)
    except Exception as e:
        raise RuntimeError(f"Error applying Gaussian Blur: {e}")


def toGrayScale(image: np.ndarray) -> np.ndarray:
    """
    Converts an image to grayscale
    :param image: Image as a numpy array
    :return: Grayscale image as a numpy array
    """
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        raise RuntimeError(f"Error converting image to grayscale: {e}")

def marrHildrethEdgeDetection(image: np.ndarray, kernelSize: int = 5, sigma: float = 1.4) -> np.ndarray:
    """
    Applies the Marr-Hildreth edge detection method using the Laplacian of Gaussian.
    :param image: Grayscale image as a numpy array
    :param kernelSize: Kernel size for Gaussian Blur
    :param sigma: Standard deviation for Gaussian Blur
    :return: Edge-detected image
    """
    try:
        smoothened = smoothen(image, kernelSize, sigma)
        edges = cv2.Laplacian(smoothened, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))  # Convert to uint8
        return edges
    except Exception as e:
        raise RuntimeError(f"Error applying Marr-Hildreth edge detection: {e}")

def sobelEdgeDetection(image: np.ndarray, dx: int = 1, dy: int = 1, ksize: int = 3) -> np.ndarray:
    """
    Applies the Sobel edge detection method.
    :param image: Grayscale image as a numpy array
    :param dx: Order of derivative x (set 1 for horizontal edges)
    :param dy: Order of derivative y (set 1 for vertical edges)
    :param ksize: Kernel size for Sobel operator
    :return: Edge-detected image
    """
    try:
        grad_x = cv2.Sobel(image, cv2.CV_64F, dx, 0, ksize=ksize)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, dy, ksize=ksize)
        
        # Compute gradient magnitude
        edges = cv2.magnitude(grad_x, grad_y)
        edges = np.uint8(edges)  # Convert to uint8
        return edges
    except Exception as e:
        raise RuntimeError(f"Error applying Sobel edge detection: {e}")

def detectEdges(image: np.ndarray, lowThreshold: int = 50, highThreshold: int = 150) -> np.ndarray:
    """
    Uses Canny edge detection to detect edges in an image
    :param image: Image as a numpy array
    :param lowThreshold: Low threshold for the edge detection
    :param highThreshold: High threshold for the edge detection
    :return: Image with edges detected
    """
    try:
        return cv2.Canny(image, lowThreshold, highThreshold)
    except Exception as e:
        raise RuntimeError(f"Error detecting edges: {e}")

def applyHoughTransform(image: np.ndarray, edges: np.ndarray) -> (np.ndarray, int):
    """
    Applies Hough Transform to detect circles in an image
    :param image: Image as a numpy array
    :param edges: Edges detected in the image
    :return: Image with circles detected, Number of coins detected
    """
    try:
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=105, param2=30, minRadius=69, maxRadius=200)
        if circles is None:
            return image.copy(), 0  # No circles detected

        output_image = image.copy()
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cv2.circle(output_image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            cv2.circle(output_image, (circle[0], circle[1]), 2, (0, 0, 255), 3)  # Center point

        return output_image, circles.shape[1]
    except Exception as e:
        raise RuntimeError(f"Error applying Hough Transform: {e}")


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

def main1() -> None:
    print("Starting Edge Detection and Coin Counting")

    try:
        # Reading the image
        image = readImage('image.jpg')

        # Converting the image to grayscale
        grayImage = toGrayScale(image)

        # Applying CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.60, tileGridSize=(12, 12))
        claheImage = clahe.apply(grayImage)

        # Smoothening the image
        smoothImage = smoothen(claheImage, 17)

        # Applying different edge detection techniques
        marrHildrethEdges = marrHildrethEdgeDetection(grayImage)
        sobelEdges = sobelEdgeDetection(grayImage)
        cannyEdges = detectEdges(smoothImage, 0, 216)
       

        # Make edges thicker using dilation for visual purposes
        kernel = np.ones((5, 5), np.uint8)
        thick_canny_edges = cv2.dilate(cannyEdges, kernel, iterations=1)
        thick_marr_edges = cv2.dilate(marrHildrethEdges, kernel, iterations=1)
        thick_sobel_edges = cv2.dilate(sobelEdges, kernel, iterations=1)

        # Create a blue mask for edges (Canny-based)
        canny_edges_colored = np.zeros_like(image)
        canny_edges_colored[:, :, 2] = thick_canny_edges  # Assign edges to the blue channel

        # Superimpose blue edges on original image
        canny_superimposed = cv2.addWeighted(image, 0.8, canny_edges_colored, 0.8, 0)

        # Performing coin counting using the grayscale image
        # Applying hough transform method
        houghTransformed, nCoins = applyHoughTransform(image, cannyEdges)
        print(f"\033[94mNumber of coins detected using Hough circle transform: {nCoins}\033[0m")

        # Saving the processed images with intuitive names
        writeImage('canny_edges_grayscale.png', thick_canny_edges)
        writeImage('canny_edges_colored.png', canny_superimposed)
        writeImage('marr_hildreth_edges.png', thick_marr_edges)
        writeImage('sobel_edges.png', thick_sobel_edges)
        writeImage('coin_detection_result_hough_transform.png', houghTransformed)
        writeImage('clahe_enhanced_image.png', claheImage)

        # Output file paths
        print(f"\033[94mInput image: file://{INPUT_DIR}image.jpg\033[0m")
        print(f"\033[94mCanny Edge Detection (Grayscale): file://{OUTPUT_DIR}canny_edges_grayscale.png\033[0m")
        print(f"\033[94mCanny Edge Detection (Superimposed): file://{OUTPUT_DIR}canny_edges_colored.png\033[0m")
        print(f"\033[94mMarr-Hildreth Edge Detection: file://{OUTPUT_DIR}marr_hildreth_edges.png\033[0m")
        print(f"\033[94mSobel Edge Detection: file://{OUTPUT_DIR}sobel_edges.png\033[0m")
        print(f"\033[94mDetected Coins Output: file://{OUTPUT_DIR}coin_detection_result.png\033[0m")
        print(f"\033[94mCLAHE Enhanced Image: file://{OUTPUT_DIR}clahe_enhanced_image.png\033[0m")
        print(f"\033[92mUse Ctrl/Cmd + Click on the above links to open the images in a new tab\033[0m")

    except Exception as e:
        print(f"\033[91mError: {e}\033[0m")

    print("Done with Edge Detection and Coin Counting")


def main2():
    print("Starting Image Segmentation and Coin Extraction")

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

    print("Done with Image Segmentation and Coin Extraction")



if __name__ == "__main__":
    main1()
    main2()