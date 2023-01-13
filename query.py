from features import partition, openIMG, extractHist
from index import normalize

def queryDatabase(query, database, color, texture, shape, weight, metric):
    """
    description:
        Query the database for the most similar images to the query.
    args:
        query: the path to the query image
        database: the database of images
        color: the color space to use
        texture: the texture descriptor to use
        shape: the shape descriptor to use
        weight: the weight of each descriptor
        metric: the metric to use
    returns:
        a dictionary of the distances between the query and the images in the database
    """
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