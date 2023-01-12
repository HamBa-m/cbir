import numpy as np
import cv2 as cv

def openIMG(filename):
    """Open an image file and return a numpy array of the image data."""
    image = cv.imread(filename)
    return image

def partition(image, v_cuts = 2, h_cuts = 2):
    """Partition the image into v_cuts * h_cuts sub images."""
    height, width, _ = image.shape
    h = int(height / v_cuts)
    w = int(width / h_cuts)
    l = []
    for i in range(v_cuts):
        for j in range(h_cuts):
            l.append(image[i*h:(i+1)*h, j*w:(j+1)*w])
    return l

def RGB(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    hist = cv.calcHist([image], channels=[0, 1, 2], mask=None,
                            histSize=[4,4,4], ranges=[0, 256] * 3)
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()


def HSV(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([image], [0, 1], None, [8, 3], [0, 180, 0, 256])
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()

def LUV(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2LUV)
    hist = cv.calcHist(images=[image], channels=[0, 1, 2], mask=None,
                            histSize=[32,32,32], ranges=[0, 256] * 3)
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten()


def LAB(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    hist = cv.calcHist(images=[image], channels=[0, 1, 2], mask=None,
                            histSize=[32,32,32], ranges=[0, 256] * 3)
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten()

def YUV(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    hist = cv.calcHist(images=[image], channels=[0, 1, 2], mask=None,
                            histSize=[32,32,32], ranges=[0, 256] * 3)
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten()

def extractHist(partition, method):
    """Extract color features from the partitioned image."""
    features = []
    for image in partition:
        features.append(method(image))
    return features

# test the functions above
if __name__ == '__main__':
    image = openIMG('database/20056.jpg')
    l = partition(image)
    iter = 0
    for e in l:
        cv.imwrite('test'+str(iter)+'.jpg', e)
        iter += 1