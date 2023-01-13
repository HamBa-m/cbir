# Content Based Image Retrival (CBIR)

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