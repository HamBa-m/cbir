import cv2 as cv
import numpy as np
from extractors import *
import glob
import sys
import colorama # for colored output in terminal
import json

def normalize(features):
    """Normalize the features of an image."""
    L= list()
    for feature in features:
        MAX = max(feature)
        MIN = min(feature)
        if MAX != MIN:
            L.append([(e-MIN)/(MAX-MIN) for e in feature])
        else:
            L.append(feature)
    return L

def makeDatabase(database, colors, textures, shapes):
    """Make a database of the images in the database folder."""
    paths = glob.glob(database + "/*.*")
    database_features = dict()
    Tmax = len(paths)
    iter = 0
    for path in paths:
        object = dict()
        img = partition(openIMG(path))
        dictionnaire = dict()
        texture_dic = dict()
        shape_dic = dict()
        for color in colors:
            dictionnaire[color.__name__] = normalize(extractHist(img, color)) # 2*4
        object["index"] = path
        object["colors"] = dictionnaire
        for texture in textures:
            texture_dic[texture.__name__] = normalize(extractHist(img, texture))
        object["textures"] = texture_dic
        for shape in shapes:    
            shape_dic[shape.__name__] = normalize(extractHist(img, shape))
        object["shapes"] = shape_dic
        database_features[path] = object
        sys.stdout.write(colorama.Fore.GREEN + "\r{1}%\t{0}>".format("="*int(50 * ((iter+1)/Tmax))+"-" *int(50 * (1 - (iter+1)/Tmax)),int(100*(iter+1)/Tmax))) 
        # progress bar
        sys.stdout.flush() 
        iter += 1
    sys.stdout.write(colorama.Fore.WHITE) # reset color of terminal
    with open("features.json", "w") as outfile:
        json.dump(database_features, outfile)

# test the function above
if __name__ == '__main__':
    print("Indexing database...")
    makeDatabase("db", [RGB, HSV, YCRCB], [LBP, GLCM, LPQ], [LOG, SOBEL,HU])