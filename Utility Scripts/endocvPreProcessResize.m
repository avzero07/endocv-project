%% Pre Processing - Resize
% @author   - Akshay Viswakumar
% @email    - akshay.viswakumar@gmail.com
% @version  - v1
% @date     - 20-March-2020

%% Details

% This script is used to resize images and masks from the original
% EndoCV2020 dataset. The contents of the dataset are of various sizes.
% This script helps resize all images to a common size before further
% processing.
%
% NOTE: The test dataset accompanying the demo has already been
% pre-processed. There is no need to run this script for the purposes of
% the demo. This has been included within the package since this was one of
% the utility scripts created as part of this project for pre-processing
% data from the original dataset.
%
% This script can be re-used to generate resized images and masks from the
% EndoCV2020 dataset or similar datasets. Please make sure to modify the
% following parts of the script to suit your dataset.
%
% 1. Replace All Paths in the 'Initialization' section with paths suitable
%    to your system environment.
%
% 2. Update the "opSize" variable in the 'Initialization' section with the
%    desired output size.
%% Implementation
clc;
clear variables
close all

%% Init

% Modify all paths according to system environment.
basePath  = "C:\Users\aksha\Documents\Projects\endocv-project\Dataset\";

valPath = basePath+"Seg-Val\";
trainPath = basePath+"Seg-Train\";
testPath = basePath+"Seg-Test\";

pathToImages = "C:\Users\aksha\Documents\Projects\endocv-project\Dataset\originalImages\";
pathToMasks = "C:\Users\aksha\Documents\Projects\endocv-project\Dataset\masks\";

opImages = basePath+"image-resized\";
opMasks = basePath+"mask-resized\";

opSize = [256 256]; % Modify to Desired Output Size
%% Get List of Filenames
imageStruct = dir(pathToImages);
imageFileName = {imageStruct.name};

maskStruct = dir(pathToMasks);
maskFileName = {maskStruct.name};

%% Loop Through Files in Image and Mask Directories

% This section runs loops that iterate over the contents of the Image and
% Mask directories. In each iteration, the image or mask is resized to the
% desired output size and then written to the output directory specified.

% Images
for i=3:length(imageFileName)
    fileName = cell2mat(imageFileName(i)) % Filename
    opIm = imresize(imread(pathToImages+fileName),opSize);
    imwrite(opIm,opImages+fileName,'png');
end

% Masks
for i=3:length(maskFileName)
    fileName = cell2mat(maskFileName(i)) % Filename
    interMask = imresize(imread(pathToMasks+fileName),opSize);
    opIm = uint8(255*mat2gray(interMask>0));
    imwrite(opIm,opMasks+fileName,'png');
end