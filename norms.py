import cv2 as cv
import numpy as np
from extractors import *
import scipy.spatial.distance as dist

def EuclidianDist(hist1, hist2):
    return dist.euclidean(hist1, hist2)

def ChiSquareDist(hist1, hist2):
    return dist.chisquare(hist1, hist2)[0]

def ManhattanDist(hist1, hist2):
    return dist.cityblock(hist1, hist2)

def CosineDist(hist1, hist2):
    return dist.cosine(hist1, hist2)

def queryDatabase(query, database, color, metric):
    """Query the database for the most similar images to the query."""
    query = partition(openIMG(query))
    query_features = dict()
    query_features = extractHist(query, color)
    distances = dict()
    for key in database:
        distances[key] = 0
        for i in range(len(database[key]["colors"][color.__name__])):
            distances[key] += metric(database[key]["colors"][color.__name__][i], query_features[i])
    return distances