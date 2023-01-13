import cv2 as cv
import numpy as np
from features import *
from index import normalize
import scipy.spatial.distance as dist

def EuclidianDist(hist1, hist2):
    """
    description: Euclidian distance between two histograms
    args:
        hist1: the first histogram
        hist2: the second histogram
    returns: 
        the Euclidian distance between the two histograms
    """
    return dist.euclidean(hist1, hist2)

def ManhattanDist(hist1, hist2):
    """
    description: Manhattan distance between two histograms
    args:
        hist1: the first histogram
        hist2: the second histogram
    returns:
        the Manhattan distance between the two histograms
    """
    return dist.cityblock(hist1, hist2)

def CosineDist(hist1, hist2):
    """
    description: Cosine distance between two histograms
    args:
        hist1: the first histogram
        hist2: the second histogram
    returns:
        the Cosine distance between the two histograms
    """
    return dist.cosine(hist1, hist2)

def similarityPercentage(target, MAX_DIST):
    """
    desceiption : Return the percentage of similarity between the query and the database.
    args:
        target: the distance between the query and the database.
        MAX_DIST: the maximum distance between the query and the database.
    returns:
        the percentage of similarity between the query and the database.
    """
    return 100*(1-target/MAX_DIST)

