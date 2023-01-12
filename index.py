import cv2 as cv
import numpy as np
from extractors import *
import glob
import sys
import colorama # for colored output in terminal
import json

def makeDatabase(database, colors):
    """Make a database of the images in the database folder."""
    paths = glob.glob(database + "/*.*")
    database_features = dict()
    Tmax = len(paths)
    iter = 0
    for path in paths:
        object = dict()
        img = partition(openIMG(path))
        dictionnaire = dict()
        for color in colors:
            dictionnaire[color.__name__] = extractHist(img, color) # 2*4
        object["index"] = path
        object["colors"] = dictionnaire
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
    makeDatabase('database', [RGB, HSV])