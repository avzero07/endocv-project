%% UNet Segmentation
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

% Init
clc
clear variables
close all

baseDir = "Dataset\";

imageDir = fullfile(baseDir,'Seg-Train\Input');
labelDir = fullfile(baseDir,'Seg-Train\Segmentation');
imageDirVal = fullfile(baseDir,'Seg-Val\Input');
labelDirVal = fullfile(baseDir,'Seg-Val\Segmentation');

%% Load Data
imds = imageDatastore(imageDir);
classNames = ["BE","Suspicious","HGD","Cancer","Polyp","Background"];
labelIDs   = [51 101 151 201 255 0];
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

imdsVal = imageDatastore(imageDirVal);
pxdsVal = pixelLabelDatastore(labelDirVal,classNames,labelIDs);

%% Create U-Net
imageSize = [256 256 3];
numClasses = 6;
encoderDepth = 3;
lgraph = unetLayers(imageSize,numClasses,'EncoderDepth',encoderDepth)

%display network
plot(lgraph)

ds = pixelLabelImageDatastore(imds,pxds);
dsVal = pixelLabelImageDatastore(imdsVal,pxdsVal);

%% Prepare to Train
%Set up training options.
q4options = trainingOptions('adam','InitialLearnRate',3e-6, 'SquaredGradientDecayFactor', 0.99,...
    'MiniBatchSize',10,'MaxEpochs',15,'VerboseFrequency',10,'Shuffle','every-epoch', ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','training-progress');

gpu1 = gpuDevice(1);

%% Train
q4net = trainNetwork(ds,lgraph,q4options);