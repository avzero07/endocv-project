%% Pre Processing - Partitioning Data
% @author   - Akshay Viswakumar
% @email    - akshay.viswakumar@gmail.com
% @version  - v1
% @date     - 20-March-2020

%% Details

% This script is used to split the original dataset into training,
% testing and validation sets. The ratio of splitting is
%
% Training Set      = 261 Images and Masks (~68% of Available Data)
% Validation Set    =  58 Images and Masks (~15% of Available Data)
% Test Set          =  67 Images and Masks (~17% of Available Data)
%
% NOTE: The Test Set made available with the demo is the complete test set
% which was paritioned from the original dataset. There is no need to run
% this script for the purposes of this demo.  This has been included within 
% the package since this was one of the utility functions created as part 
% of this project for pre-processing masks from the original dataset.
%
% This script can be re-used to generate composite masks for data from the
% EndoCV2020 dataset or similar datasets. Please make sure to modify the
% following parts of the script to suit your dataset.
%
% 1. Replace All Paths in the 'Initialization' section with paths suitable
%    to your system environment.

%% Implementation
clc;
clear variables
close all
%% Initialization

% Modify all paths according to system environment.
basePath = "C:\Users\aksha\Documents\Projects\endocv-project\Dataset\";
valPath = basePath+"Seg-Val\";
trainPath = basePath+"Seg-Train\";
testPath = basePath+"Seg-Test\";

pathToImages = basePath+"image-resized\"; 
pathToMasks = basePath+"mask-resized\";
pathOP = basePath+"mask-resized-combined\";

%% Get List of Filenames

imageStruct = dir(pathToImages);
imageFileName = {imageStruct.name}; % List of Image Filenames

%% Split Data into Train, Test and Validate

%% 1. Split Data into Train and Test Sets
rng(0,'twister'); % For Repeatability

r = randi([1 386],1,77);

% Split Data into Train / Test
for i=3:length(imageFileName)
    fileName = cell2mat(imageFileName(i));
    if(ismember(i-2,r))
        i
        copyfile(pathToImages+fileName,testPath+'Input\');
        copyfile(pathOP+fileName,testPath+'Segmentation\')
    else
        copyfile(pathToImages+fileName,trainPath+'Input\');
        copyfile(pathOP+fileName,trainPath+'Segmentation\')
    end
end

%% 2. Split Train Set into Training and Validation Sets
trainFileNameStruct = dir(trainPath+'Input\');
trainFileName = {trainFileNameStruct.name};

rng(0,'twister'); % For Repeatability
r = randi([1 386],1,77);
r2 = randi([1 309],1,62);

for i=3:length(trainFileName)
    fileName = cell2mat(trainFileName(i));
    
    if(ismember(i-2,r2))
        copyfile(trainPath+'Input\'+fileName,valPath+'Input\');
        delete(trainPath+'Input\'+fileName)
        copyfile(trainPath+'Segmentation\'+fileName,valPath+'Segmentation\');
        delete(trainPath+'Segmentation\'+fileName)
    end
end