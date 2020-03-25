# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:16:48 2020

@author: akshay
"""

# Imports
import os
import json
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# Init Dict
metaData = {}

# Path to Bounding Boxes
path = "C:/Users/aksha/Documents/Github Projects/endocv-project/Dataset/bbox/"


for file in os.listdir(path):
    filename = os.fsdecode(file)
    
    with open(path+filename) as f:
        metaData[filename[:-4]] = {}
        index = 0
        for line in f:
            if(line != ''):
                lineList = line.split()
                metaData[filename[:-4]][str(index)] = {}
                metaData[filename[:-4]][str(index)]['left'] = lineList[0] 
                metaData[filename[:-4]][str(index)]['top'] = lineList[1]
                metaData[filename[:-4]][str(index)]['right'] = lineList[2]
                metaData[filename[:-4]][str(index)]['bottom'] = lineList[3]
                metaData[filename[:-4]][str(index)]['class'] = lineList[4]
                index = index + 1

with open("C:/Users/aksha/Documents/Github Projects/endocv-project/Dataset/bbox-json.txt","w") as file:
    json.dump(metaData, file)

with open("C:/Users/aksha/Documents/Github Projects/endocv-project/Dataset/bbox-json.txt","r") as file:
    temp = json.load(file)
    
targetW = 343
targetH = 356

# Loop Through Images
imgDir = "C:/Users/aksha/Documents/Github Projects/endocv-project/Dataset/originalImages/"

index = 0
for imgFile in os.listdir(imgDir):
    imgFileName = os.fsdecode(imgFile)
    data = image.imread(imgDir+imgFileName)
    y = data.shape[0] # Width
    yScale = targetW/y
    x = data.shape[1] # Height
    xScale = targetH/x
    
    # Loop Through Keys of Metadata and Update Bounding Boxes
    for key in metaData[imgFileName[:-4]]:
        oldLeft =  float(metaData[imgFileName[:-4]][key]['left'])
        metaData[imgFileName[:-4]][key]['left'] = str(int(np.round(oldLeft*xScale)))
        oldBottom = float(metaData[imgFileName[:-4]][key]['bottom'])
        metaData[imgFileName[:-4]][key]['bottom'] = str(int(np.round(oldBottom*yScale)))
        oldRight = float(metaData[imgFileName[:-4]][key]['right'])
        metaData[imgFileName[:-4]][key]['right'] = str(int(np.round(oldRight*xScale)))
        oldTop = float(metaData[imgFileName[:-4]][key]['top'])
        metaData[imgFileName[:-4]][key]['top'] = str(int(np.round(oldTop*yScale)))

with open("C:/Users/aksha/Documents/Github Projects/endocv-project/Dataset/bbox-json-resized.txt","w") as file:
    json.dump(metaData, file)
    
# Plot Image

imgFileName = 'EDD2020_AJ0062.jpg'

# Load Image
img = image.imread("C:/Users/aksha/Documents/Github Projects/endocv-project/Dataset/originalImages-resized/"+imgFileName)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(img)

# Create a Rectangle patch
left = int(metaData[imgFileName[:-4]][key]['left'])
top = int(metaData[imgFileName[:-4]][key]['top'])
right = int(metaData[imgFileName[:-4]][key]['right'])
bottom = int(metaData[imgFileName[:-4]][key]['bottom'])
rect = patches.Rectangle((left,top),bottom-top,right-left,linewidth=1,edgecolor='r',facecolor='none')
#rect = patches.Rectangle((199,441),830-441,622-199,linewidth=1,edgecolor='r',facecolor='none')
# Add the patch to the Axes
ax.add_patch(rect)

plt.show()