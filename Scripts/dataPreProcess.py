# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:14:25 2020

@author: akshay
"""

# Script to Preprocess Data

# Import Statements
import PIL
import numpy as np
import os
from matplotlib import image
from matplotlib import pyplot

def scopeImages(inpDirectory,numFiles,imgExtension):
    # Getting File Size
    imWidths = np.zeros([numFiles,1])
    imHeights = np.zeros([numFiles,1])
    imChannels = np.zeros([numFiles,1])
    imName = list()

    directory = os.fsencode(inpDirectory)

    index = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(imgExtension): 
            data = image.imread(os.fsdecode(directory+file))
            imWidths[index,0] = data.shape[1]
            imHeights[index,0] = data.shape[0]
            #imChannels[index,0] = data.shape[2]
            imName.append(filename)
            index = index + 1
            continue
        else:
            continue
        
    # Print Stats
    print("Smallest Width")
    print(np.min(imWidths))
    print(np.argmin(imWidths))
    print(imName[np.argmin(imWidths)])
    
    print("Smallest Height")
    print(np.min(imHeights))
    print(np.argmin(imHeights))
    print(imName[np.argmin(imHeights)])
    
    return imWidths, imHeights, imChannels, imName

def modImages(targetWidth,targetHeight,inputDirectory,targetDirectory,imgExtension):
    # Target Resolution
    print("Target Resolution = {} x {} (width by height)".format(targetWidth,targetHeight))
    
    # Rescale and Save Images
    
    # Run Loop to Resize
    index = 0
    for file in os.listdir(inputDirectory):
        filename = os.fsdecode(file)
        if filename.endswith(imgExtension): 
            img = PIL.Image.open(os.fsdecode(inputDirectory+file))
            resized = img.resize((targetWidth,targetHeight))
            resized.save(targetDirectory+filename,format='JPEG')
            index = index + 1
            continue
        else:
            continue
        
inpDir = "C:/Users/aksha/Documents/Github Projects/endocv-project/Dataset/originalImages/"
targDir = "C:/Users/aksha/Documents/Github Projects/endocv-project/Dataset/originalImages-resized-2/"

inpDirMask = "C:/Users/aksha/Documents/Github Projects/endocv-project/Dataset/masks/"
targDirMask = "C:/Users/aksha/Documents/Github Projects/endocv-project/Dataset/masks-resized-2/"

# Images
imWidths, imHeights, imChannels, imName = scopeImages(inpDir,386,'jpg')
modImages(256,256,inpDir,targDir,'jpg')

# Masks
imWidths, imHeights, imChannels, imName = scopeImages(inpDirMask,502,'tif')
modImages(256,256,inpDirMask,targDirMask,'tif')