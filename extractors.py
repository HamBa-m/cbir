import numpy as np
import cv2 as cv
import pywt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

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


def YCRCB(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    hist = cv.calcHist(images=[image], channels=[0, 1, 2], mask=None,
                            histSize=[4,4,4], ranges=[0, 256, 0, 256, 0, 256])
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()

def LBP(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp, density=True, bins=10, range=(0, 10))
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()

def GLCM(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    glcm = graycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    ASM = graycoprops(glcm, 'ASM')
    features = [float(x) for x in contrast.ravel()]
    features.extend([float(x) for x in dissimilarity.ravel()])
    features.extend([float(x) for x in homogeneity.ravel()])
    features.extend([float(x) for x in energy.ravel()])
    features.extend([float(x) for x in correlation.ravel()])
    features.extend([float(x) for x in ASM.ravel()])
    return features

def MRF(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mrf = cv.cornerHarris(image, 2, 3, 0.04)
    hist, _ = np.histogram(mrf, density=True, bins=10, range=(0, 10))
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()

def LPQ(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    lpq = cv.Laplacian(image, cv.CV_8UC1, ksize=3)
    hist, _ = np.histogram(lpq, density=True, bins=10, range=(0, 10))
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()

def HU(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hu = cv.HuMoments(cv.moments(image)).flatten()
    return hu.tolist()

def LOG(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    log = cv.Laplacian(image, cv.CV_8UC1, ksize=3)
    hist, _ = np.histogram(log, density=True, bins=10, range=(0, 10))
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()

def SOBEL(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sobel = cv.Sobel(image, cv.CV_8UC1, 1, 0, ksize=3)
    hist, _ = np.histogram(sobel, density=True, bins=10, range=(0, 10))
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()

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