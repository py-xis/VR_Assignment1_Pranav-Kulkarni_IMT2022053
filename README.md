# VR_Assignment1_Pranav-Kulkarni_IMT2022053

<p align="center">
    <h1>Visual Recognition - Assignment 1</h1>
</p>

## Overview
This project implements an image processing pipeline for:
- **Edge Detection** (Canny, Sobel, Marr-Hildreth)
- **Image Segmentation** (Thresholding, Watershed Algorithm)
- **Coin Counting** (Contour Detection, Hough Circle Transform)
- **Image Stitching** (SIFT, Homography, Warping and Blending)

## Folder Structure
```
.
├── Part 1
│   ├── code
│   │   ├── detect.py
│   │   └── segment.py
│   ├── input
│   │   └── image.jpg
│   └── results
│       ├── canny_edges_colored.png
│       ├── canny_edges_grayscale.png
│       ├── clahe_enhanced_image.png
│       ├── coin_detection_result_contour_transform.png
│       ├── coin_detection_result_hough_transform.png
│       ├── marr_hildreth_edges.png
│       ├── original_image_histogram.png
│       ├── segmentation
│       │   ├── adaptive_threshold.png
│       │   ├── coins
│       │   │   ├── coin_1.jpg
│       │   │   ├── coin_10.jpg
│       │   │   ├── coin_2.jpg
│       │   │   ├── coin_3.jpg
│       │   │   ├── coin_4.jpg
│       │   │   ├── coin_5.jpg
│       │   │   ├── coin_6.jpg
│       │   │   ├── coin_7.jpg
│       │   │   ├── coin_8.jpg
│       │   │   └── coin_9.jpg
│       │   ├── manual_threshold.png
│       │   ├── otsu_threshold.png
│       │   └── segmented_image.png
│       └── sobel_edges.png
├── Part 2
│   ├── code
│   │   └── image_stitching.py
│   ├── input
│   │   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   └── image_3.jpg
│   └── output
│       └── stitched.png
├── README.md
└── VR_Assignment1_Pranav Kulkarni_IMT2022053.pdf
```

## Dependencies
The following Python libraries are required:
- `numpy`
- `matplotlib`
- `opencv-python`

Install them using:
```bash
pip install numpy matplotlib opencv-python
```

## Running the Code

### Part 1: Edge Detection, Segmentation, and Coin Counting
Navigate to the **Part 1/code/** directory and run:
```bash
python detect.py
```
This will process `image.jpg` and generate results in `Part 1/results/`.

For segmentation:
```bash
python segment.py
```

### Part 2: Image Stitching
Navigate to **Part 2/code/** and run:
```bash
python image_stitching.py
```
This will stitch the images from `Part 2/input/` and save the output in `Part 2/output/stitched.png`.

## Results
- Processed images are stored in `Part 1/results/`.
- Segmented images are inside `Part 1/results/segmentation/`.
- Extracted coin images are inside `Part 1/results/segmentation/coins/`.
- The final stitched panorama is in `Part 2/output/`.

## Report
The complete report can be found in:
```
VR_Assignment1_Pranav Kulkarni_IMT2022053.pdf
