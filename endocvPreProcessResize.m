%% Pre Processing - Resize
% @author   - Akshay Viswakumar
% @email    - akshay.viswakumar@gmail.com
% @version  - v0.5
% @date     - 31-March-2020
%% Changelog
% Version 0.5
% -- Initial Implementation
%
% To Do
% -----
% -- 
%% Implementation
clc;
clear variables
close all

%% Init

basePath  = "C:\Users\aksha\Documents\Github Projects\endocv-project\Dataset\";

valPath = basePath+"Seg-Val\";
trainPath = basePath+"Seg-Train\";
testPath = basePath+"Seg-Test\";

pathToImages = "C:\Users\aksha\Documents\Github Projects\endocv-project\Dataset\originalImages\";
pathToMasks = "C:\Users\aksha\Documents\Github Projects\endocv-project\Dataset\masks\";

opImages = basePath+"image-resized\";
opMasks = basePath+"mask-resized\";

opSize = [256 256];
%% Get List of Filenames
imageStruct = dir(pathToImages);
imageFileName = {imageStruct.name};

maskStruct = dir(pathToMasks);
maskFileName = {maskStruct.name};

%% Loop Through Files

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