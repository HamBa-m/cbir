from index import *
from metrics import *
from features import *

# load the database
with open("features.json", "r") as infile:
    database = json.load(infile)

# open the query image
query = "20485.jpg"

# query the database for the most similar images
distances = queryDatabase(query, database, YCRCB, LPQ, HU,[1,0,0], EuclidianDist)

# sort the distances
distances = sorted(distances.items(), key=lambda x: x[1])

# print the 5 most similar images
for i in range(20):
    print(distances[i], similarityPercentage(distances[i][1], distances[-1][1]))