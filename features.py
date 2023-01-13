import numpy as np
import cv2 as cv
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def openIMG(filename):
    """
    description:
        Open an image.
    args:
        filename: the path of the image.
    returns:
        the image.
    """
    image = cv.imread(filename)
    return image

def partition(image, v_cuts = 2, h_cuts = 2):
    """
    description:
        Partition an image into subimages.
    args:
        image: the image to partition.
        v_cuts: the number of vertical cuts.
        h_cuts: the number of horizontal cuts.
    returns:
        a list of subimages.
    """
    height, width, _ = image.shape
    h = int(height / v_cuts)
    w = int(width / h_cuts)
    l = []
    for i in range(v_cuts):
        for j in range(h_cuts):
            l.append(image[i*h:(i+1)*h, j*w:(j+1)*w])
    return l

def RGB(image):
    """
    description:
        Extract the RGB histogram of an image, which stands for Red, Green, Blue.
    args:
        image: the image to extract the histogram from.
    returns:   
        the RGB histogram of the image.
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    hist = cv.calcHist([image], channels=[0, 1, 2], mask=None,
                            histSize=[4,4,4], ranges=[0, 256] * 3)
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()

def HSV(image):
    """
    description:
        Extract the HSV histogram of an image, which stands for Hue, Saturation, Value.
    args:
        image: the image to extract the histogram from.
    returns:   
        the HSV histogram of the image.
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([image], [0, 1], None, [8, 3], [0, 180, 0, 256])
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()

def YCRCB(image):
    """
    description:
        Extract the YCrCb histogram of an image, which stands for Y Chrominance, Cr Chrominance, Cb Chrominance.
    args:
        image: the image to extract the histogram from.
    returns:   
        the YCrCb histogram of the image.
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    hist = cv.calcHist(images=[image], channels=[0, 1, 2], mask=None,
                            histSize=[4,4,4], ranges=[0, 256, 0, 256, 0, 256])
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()

def LBP(image):
    """
    description:
        Extract the LBP histogram of an image, which stands for Local Binary Pattern.
    args:
        image: the image to extract the histogram from.
    returns:   
        the LBP histogram of the image.
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp, density=True, bins=10, range=(0, 10))
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()

def GLCM(image):
    """
    description:
        Extract the GLCM features of an image, which stands for Gray Level Co-occurrence Matrix.
    args:
        image: the image to extract the features from.
    returns:
        the GLCM features of the image, in the following order:
        contrast, dissimilarity, homogeneity, energy, correlation, ASM.
    """
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
    """
    description:
        Extract the MRF histogram of an image, which stands for the Markov Random Field.
    args:
        image: the image to extract the histogram from.
    returns:
        the MRF histogram of the image
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mrf = cv.cornerHarris(image, 2, 3, 0.04)
    hist, _ = np.histogram(mrf, density=True, bins=10, range=(0, 10))
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()

def LPQ(image):
    """
    description:
        Extract the LPQ histogram of an image, which stands for Laplacian Pyramid Quantization.
    args:
        image: the image to extract the histogram from.
    returns:
        the LPQ histogram of the image
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    lpq = cv.Laplacian(image, cv.CV_8UC1, ksize=3)
    hist, _ = np.histogram(lpq, density=True, bins=10, range=(0, 10))
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()

def HU(image):
    """
    description:
        Extract the Hu moments of an image, which stands for Hu invariants.
    args:
        image: the image to extract the moments from.
    returns:
        the Hu moments of the image
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hu = cv.HuMoments(cv.moments(image)).flatten()
    return hu.tolist()

def LOG(image):
    """
    description:
        Extract the LOG histogram of an image, which stands for Laplacian of Gaussian.
    args:
        image: the image to extract the histogram from.
    returns:
        the LOG histogram of the image
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    log = cv.Laplacian(image, cv.CV_8UC1, ksize=3)
    hist, _ = np.histogram(log, density=True, bins=10, range=(0, 10))
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()

def SOBEL(image):
    """
    description:
        Extract the SOBEL histogram of an image, which stands for Sobel operator.
    args:
        image: the image to extract the histogram from.
    returns:
        the SOBEL histogram of the image
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sobel = cv.Sobel(image, cv.CV_8UC1, 1, 0, ksize=3)
    hist, _ = np.histogram(sobel, density=True, bins=10, range=(0, 10))
    hist = cv.normalize(hist, dst=hist.shape)
    return hist.flatten().tolist()

def extractHist(partition, method):
    """
    description:
        Extract the histogram of an image partition using a given method.
    args:
        partition: the image partition to extract the histogram from.
        method: the method to extract the histogram.
    returns:
        the histogram of the image partition.
    """
    features = []
    for image in partition:
        features.append(method(image))
    return features

# test the functions above
# if __name__ == '__main__':
#     image = openIMG('database/20056.jpg')
#     l = partition(image)
#     iter = 0
#     for e in l:
#         cv.imwrite('test'+str(iter)+'.jpg', e)
#         iter += 1