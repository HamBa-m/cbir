import cv2 as cv
import numpy as np
from extractors import *
from index import normalize
import scipy.spatial.distance as dist

def EuclidianDist(hist1, hist2):
    return dist.euclidean(hist1, hist2)

def ManhattanDist(hist1, hist2):
    return dist.cityblock(hist1, hist2)

def CosineDist(hist1, hist2):
    return dist.cosine(hist1, hist2)

def queryDatabase(query, database, color, texture, shape, weight, metric):
    """Query the database for the most similar images to the query."""
    query = partition(openIMG(query))
    query_features = dict()
    query_features["Color"] = normalize(extractHist(query, color))
    query_features["Texture"] = normalize(extractHist(query, texture))
    query_features["Shape"] = normalize(extractHist(query, shape))
    distances = dict()
    for key in database:
        distances[key] = 0
        for i in range(len(database[key]["colors"][color.__name__])):
            distances[key] += weight[0]*metric(database[key]["colors"][color.__name__][i], query_features["Color"][i])
            distances[key] += weight[1]*metric(database[key]["textures"][texture.__name__][i], query_features["Texture"][i])
            distances[key] += weight[2]*metric(database[key]["shapes"][shape.__name__][i], query_features["Shape"][i])
    return distances

def similarityPercentage(target, MAX_DIST):
    """Return the percentage of similarity between the query and the database."""
    return 100*(1-target/MAX_DIST)

