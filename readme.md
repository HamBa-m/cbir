# Content Based Image Retrival (CBIR)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Python Version](https://img.shields.io/badge/Python-3.x-green.svg)](https://www.python.org/downloads/) [![Project Version](https://img.shields.io/badge/Project%20Version-1.0-lightgrey.svg)](https://github.com/HamzaBamohammed/CBIR)

## Introduction

CBIR is a computer vision technique for retrieving images from a database based on the similarity of their content. The content of an image is usually described by a set of features. The features are extracted from the image and then compared to the features of the images in the database. The images with the most similar features are returned as the result of the search. The features can be color, texture, shape, etc. The most common features are color, texture, and shape. The features are extracted from the image and then compared to the features of the images in the database. The images with the most similar features are returned as the result of the search. The features can be color, texture, and shape.

## Features Extraction
For each image in the database, we extract the features and store them in a JSON file ```features.json```. The features are extracted using the following methods:

### by color space


- RGB: Red, Green, Blue. 
- HSV: Hue, Saturation, Value.
- YCrCb: Luminance, Chrominance Red, Chrominance Blue.

### by texture

- LBP: Local Binary Pattern.
- GLCM: Gray Level Co-occurrence Matrix.
- LPQ: Local Phase Quantization.

### by shape

- LOG: Laplacian of Gaussian.
- SOBEL: Sobel operator.
- HU: Hu moments.

## Query Image
The query image is the image that we want to search for in the database. The query image is also extracted using the same methods as the database images. The query image is then compared to the images in the database using the features and a metric is calculated for each image in the database. The images are then sorted based on the metric and the images with the lowest distance are returned as the result of the search.

## Metrics for Features Comparison
The features are matched using the following metrics:

- Euclidean distance.
- Cosine similarity.
- Manhattan distance.

The similarity score is calculated for each image in the database and the images are sorted based on the similarity score. The images with the highest similarity score are returned as the result of the search.

## Authors
- [@FilaliHicham](https://www.github.com/FILALIHicham), Applied Mathematics & AI engineering student at ENSIAS.
- [@HamzaBamohammed](https://www.github.com/HamzaBamohammed), Applied Mathematics & AI engineering student at ENSIAS
- [@BouchraSahri](https://www.github.com/bouchrasa), Applied Mathematics & AI engineering student at ENSIAS

## Instructions
1. Clone the repository to your local machine
```bash
git clone https://github.com/HamzaBamohammed/cbir
```
2. Install the required dependencies
```bash
pip install -r requirements.txt
```
3. Change the images in the database folder ```database``` with your own images if needed.
4. Run the indexation script, this will extract the features of the images in the database and store them in a JSON file ```features.json```
```bash
python index.py
```
5. Run the main script, that will open the GUI and you can load the query image and search for similar images in the database.
```bash
python main.py
```

## Example

1. When the main script is run, the following GUI will open.
![GUI](https://archive.org/download/screenshot-4_202301/screenshot%201.png)

2. Load the query image by clicking on the ```Load Query Image``` button, then select the image from the file explorer (not necessarily from the database).
![Load Query](https://archive.org/download/screenshot-4_202301/screenshot%202.png)

3. Select the combination of the color space, texture, and shape descriptors that you want to use for the search, and then click the search button.
![Search](https://archive.org/download/screenshot-4_202301/screenshot%203.png)

4. The result of the search will be displayed in the GUI.
![Result](https://archive.org/download/screenshot-4_202301/screenshot%204.png)
