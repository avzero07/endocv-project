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

imageDirTest = fullfile(baseDir,'Seg-Test\Input');
labelDirTest = fullfile(baseDir,'Seg-Test\Segmentation');

outputPath = fullfile(baseDir,'Seg-Output\');

%% Load Data
imds = imageDatastore(imageDir);
classNames = ["BE","Suspicious","HGD","Cancer","Polyp","Background"];
labelIDs   = [51 101 151 201 255 0];
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

imdsVal = imageDatastore(imageDirVal);
pxdsVal = pixelLabelDatastore(labelDirVal,classNames,labelIDs);

imdsTest = imageDatastore(imageDirTest);
pxdsTest = pixelLabelDatastore(labelDirTest,classNames,labelIDs);

%% Visualize Data

tbl = countEachLabel(pxds);
frequency = tbl.PixelCount/sum(tbl.PixelCount);

figure
bar(1:numel(classNames),frequency)
xticks(1:numel(classNames)) 
xticklabels(tbl.Name)
xtickangle(45)
grid on
title("Histogram of Pixels")
ylabel('Frequency')

%% Balancing Weights
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);

%% Data Augmentation Code
augmenter = imageDataAugmenter('RandXReflection',true,'RandYReflection',true,'RandRotation',[0 360]);

%% Create U-Net
imageSize = [256 256 3];
numClasses = 6;
encoderDepth = 4;
lgraph = unetLayers(imageSize,numClasses,'EncoderDepth',encoderDepth)

% Replace Classification Layer
lgraph = replaceLayer(lgraph,"Segmentation-Layer",pxLayer);

%display network
plot(lgraph)

ds = pixelLabelImageDatastore(imds,pxds,'DataAugmentation',augmenter);
dsVal = pixelLabelImageDatastore(imdsVal,pxdsVal);

%% Prepare to Train
%Set up training options.
q4options = trainingOptions('adam','InitialLearnRate',3e-6, 'SquaredGradientDecayFactor', 0.99,'ValidationPatience',7,...
    'MiniBatchSize',10,'MaxEpochs',100,'VerboseFrequency',10,'Shuffle','every-epoch', ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','training-progress','CheckpointPath','Checkpoints\1000-Trial-1-PreBuilt\');

gpu1 = gpuDevice(1);

%% Train
myUnet = trainNetwork(ds,lgraph,q4options);

%% Evaluate

pxdsResults = semanticseg(imdsTest,net,'MiniBatchSize',4,'WriteLocation',outputPath,'Verbose',true);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',true);

%% Test on 1 Image

num = 10;
tI = readimage(imdsTest,num);
tS = semanticseg(tI, myUnet);
tG = readimage(pxdsTest,num);

cmap = endocvCmap();

B = labeloverlay(tI,tS,'Colormap',cmap,'Transparency',0.4);
figure, imshow(B)
pixelLabelColorbar(cmap,classNames);
title('Predicted Output')

C = labeloverlay(tI,tG,'Colormap',cmap,'Transparency',0.4);
figure, imshow(C)
pixelLabelColorbar(cmap,classNames);
title('Ground Truth')

%% Load and Resume Training
%load('Checkpoints\Trial Run\net_checkpoint__104__2020_04_13__09_36_39.mat') % Loads Network as 'net'
%plot(layerGraph(net)) % Plot Loaded Network

%% Resume Training
%myUnetResume = trainNetwork(ds,layerGraph(net),q4options);