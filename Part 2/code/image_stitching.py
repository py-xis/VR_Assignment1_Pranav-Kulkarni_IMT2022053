import cv2
import numpy as np

INPUT_DIR = '../input/'
OUTPUT_DIR = '../output/'

def extractKeyPoints(image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray, np.ndarray]:
    """
    Extract SIFT keypoints and descriptors from the image.
    Returns:
        kp: list of keypoints.
        des: descriptor array.
        keypointsImg: image with keypoints drawn (for visualization).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keyPoints, descriptors = sift.detectAndCompute(gray, None)
    keypointsImg = cv2.drawKeypoints(
        image, keyPoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return keyPoints, descriptors, keypointsImg

def matchKeyPoints(descriptors1: np.ndarray, descriptors2: np.ndarray, ratio: float = 0.75) -> list[cv2.DMatch]:
    """
    Match SIFT descriptors between two images using BFMatcher + Lowe's ratio test.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    rawMatches = bf.knnMatch(descriptors1, descriptors2, k=2)
    goodMatches = []
    for match1, match2 in rawMatches:
        if match1.distance < ratio * match2.distance:
            goodMatches.append(match1)
    return goodMatches

def getHomography(keyPoints1: list[cv2.KeyPoint], keyPoints2: list[cv2.KeyPoint], matches: list[cv2.DMatch], reprojThresh: float = 4.0) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Compute homography from matched keypoints.
    """
    if len(matches) < 4:
        return None, None
    
    pointsA = np.float32([keyPoints1[m.queryIdx].pt for m in matches])
    pointsB = np.float32([keyPoints2[m.trainIdx].pt for m in matches])
    homographyMatrix, status = cv2.findHomography(pointsB, pointsA, cv2.RANSAC, reprojThresh)
    return homographyMatrix, status

def warpAndBlend(imageA: np.ndarray, imageB: np.ndarray, homographyMatrix: np.ndarray) -> np.ndarray:
    """
    Warp imageB into imageA's plane using the homography matrix and blend overlaps.
    """
    heightA, widthA = imageA.shape[:2]
    heightB, widthB = imageB.shape[:2]

    cornersB = np.float32([[0, 0], [0, heightB], [widthB, heightB], [widthB, 0]]).reshape(-1, 1, 2)
    warpedCornersB = cv2.perspectiveTransform(cornersB, homographyMatrix)
    
    cornersA = np.float32([[0, 0], [0, heightA], [widthA, heightA], [widthA, 0]]).reshape(-1, 1, 2)
    allCorners = np.concatenate((cornersA, warpedCornersB), axis=0)

    [xmin, ymin] = np.int32(allCorners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(allCorners.max(axis=0).ravel() + 0.5)
    
    translationX, translationY = -xmin, -ymin
    translationMatrix = np.array([[1, 0, translationX], [0, 1, translationY], [0, 0, 1]], dtype=np.float32)

    panoramaWidth = xmax - xmin
    panoramaHeight = ymax - ymin
    panoramaB = cv2.warpPerspective(imageB, translationMatrix.dot(homographyMatrix), (panoramaWidth, panoramaHeight))
    
    panorama = np.zeros((panoramaHeight, panoramaWidth, 3), dtype=np.uint8)
    panorama[translationY:translationY + heightA, translationX:translationX + widthA] = imageA

    maskA = (panorama.sum(axis=2) > 0).astype(np.uint8)
    maskB = (panoramaB.sum(axis=2) > 0).astype(np.uint8)
    overlapMask = (maskA & maskB)

    onlyAIdx = np.where((maskA == 1) & (maskB == 0))
    onlyBIdx = np.where((maskA == 0) & (maskB == 1))
    overlapIdx = np.where(overlapMask == 1)

    blended = np.zeros_like(panorama)
    blended[onlyAIdx] = panorama[onlyAIdx]
    blended[onlyBIdx] = panoramaB[onlyBIdx]

    alpha = 0.5
    blended[overlapIdx] = (alpha * panorama[overlapIdx] + (1 - alpha) * panoramaB[overlapIdx]).astype(np.uint8)

    gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return blended
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    finalPanorama = blended[y:y + h, x:x + w]

    return finalPanorama

def stitchPair(imageA: np.ndarray, imageB: np.ndarray) -> np.ndarray | None:
    """
    Compute the homography between imageA and imageB, then warp and blend them.
    """
    keyPointsA, descriptorsA, _ = extractKeyPoints(imageA)
    keyPointsB, descriptorsB, _ = extractKeyPoints(imageB)
    
    matches = matchKeyPoints(descriptorsA, descriptorsB, ratio=0.75)
    if len(matches) < 4:
        print("Not enough matches to stitch these images.")
        return None
    
    homographyMatrix, _ = getHomography(keyPointsA, keyPointsB, matches)
    if homographyMatrix is None:
        print("Could not compute a valid homography.")
        return None
    
    return warpAndBlend(imageA, imageB, homographyMatrix)


if __name__ == '__main__':
    print("Starting")
    image1 = cv2.imread(INPUT_DIR + 'image_1.jpg')
    image2 = cv2.imread(INPUT_DIR + 'image_2.jpg')
    image3 = cv2.imread(INPUT_DIR + 'image_3.jpg')
    
    pano12 = stitchPair(image1, image2)
    if pano12 is None:
        print("Failed to stitch image1 and image2.")
    
    finalPanorama = stitchPair(pano12, image3)
    if finalPanorama is None:
        print("Failed to stitch the panorama with image3.")
    
    cv2.imwrite(OUTPUT_DIR + 'stitched.png', finalPanorama)
    print(f"\033[94mStiched Image: file://{OUTPUT_DIR}/stitched.png\033[0m")
    print(f"\033[92mUse Ctrl/Cmd + Click on the above links to open the images in a new tab\033[0m")

    print("Done")