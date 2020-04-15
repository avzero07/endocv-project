%% Manual VGG16 UNet
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
clc;
clear variables;
close all;

% Reference
load('Checkpoints\Trial Run\net_checkpoint__104__2020_04_13__09_48_42.mat')

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

tbl = countEachLabel(pxds);
frequency = tbl.PixelCount/sum(tbl.PixelCount);

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;

%% Load VGG16

VGGnet = vgg16;
lgraphVGG = layerGraph(VGGnet.Layers(1:32)); % Extract Upto Start of Dense

plot(lgraphVGG)

%% Assign Layers

imageSize = [256 256 3];
numClasses = length(classNames);

layers = [
    
    % Bridge Stage
    
    % Bridge Stage Convolution
    convolution2dLayer(3,1024,'Padding','same','Name','Bridge-Stage-Conv')
    reluLayer('Name','Relu_Bridge-1')
    
    % Bridge Stage UpConvolution
    transposedConv2dLayer(2,512,'Stride',[2 2],'Cropping','same','Name','Bridge-Stage-Trans_Conv')
    
    % Decoder Stage 1
    
    % Decoder Stage 1 - Depth Concatenation
    depthConcatenationLayer(2,'Name','Decoder-Stage-1-DepthConcat')
    
    % Decoder Stage 1 - Convolution #1
    convolution2dLayer(3,512,'Padding','same','Name','Decoder-Stage-1-Conv_1')
    reluLayer('Name','Relu_5-1')
    
    % Decoder Stage 1 - Convolution #2
    convolution2dLayer(3,512,'Padding','same','Name','Decoder-Stage-1-Conv_2')
    reluLayer('Name','Relu_5-2')
    
    % Decoder Stage 1 - Transpose Convolution
    transposedConv2dLayer(2,256,'Stride',[2 2],'Cropping','same','Name','Decoder-Stage-1-Trans_Conv')
    
    % Decoder Stage 2
    
    % Decoder Stage 2 - Depth Concatenation
    depthConcatenationLayer(2,'Name','Decoder-Stage-2-DepthConcat')
    
    % Decoder Stage 2 - Convolution #1
    convolution2dLayer(3,256,'Padding','same','Name','Decoder-Stage-2-Conv_1')
    reluLayer('Name','Relu_6-1')
    
    % Decoder Stage 2 - Convolution #2
    convolution2dLayer(3,256,'Padding','same','Name','Decoder-Stage-2-Conv_2')
    reluLayer('Name','Relu_6-2')
    
    % Decoder Stage 2 - Transpose Convolution
    transposedConv2dLayer(2,128,'Stride',[2 2],'Cropping','same','Name','Decoder-Stage-2-Trans_Conv')
    
    % Decoder Stage 3
    
    % Decoder Stage 3 - Depth Concatenation
    depthConcatenationLayer(2,'Name','Decoder-Stage-3-DepthConcat')
    
    % Decoder Stage 3 - Convolution #1
    convolution2dLayer(3,128,'Padding','same','Name','Decoder-Stage-3-Conv_1')
    reluLayer('Name','Relu_7-1')
    
    % Decoder Stage 3 - Convolution #2
    convolution2dLayer(3,128,'Padding','same','Name','Decoder-Stage-3-Conv_2')
    reluLayer('Name','Relu_7-2')
    
    % Decoder Stage 3 - Transpose Convolution
    transposedConv2dLayer(2,64,'Stride',[2 2],'Cropping','same','Name','Decoder-Stage-3-Trans_Conv')
    
    % Decoder Stage 4
    
    % Decoder Stage 4 - Depth Concatenation
    depthConcatenationLayer(2,'Name','Decoder-Stage-4-DepthConcat')
    
    % Decoder Stage 4 - Convolution #1
    convolution2dLayer(3,64,'Padding','same','Name','Decoder-Stage-4-Conv_1')
    reluLayer('Name','Relu_8-1')
    
    % Decoder Stage 4 - Convolution #2
    convolution2dLayer(3,64,'Padding','same','Name','Decoder-Stage-4-Conv_2')
    reluLayer('Name','Relu_8-2')
    
    % Output
    convolution2dLayer(1,numClasses,'Name','Final Convolution','Padding','same')
    softmaxLayer('Name','SoftMax')
    pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights)
    
    ];

lgraph = addLayers(lgraphVGG,layers);
lgraph = connectLayers(lgraph,'pool5','Bridge-Stage-Conv');

% Skip Connections
lgraph = connectLayers(lgraph,'relu4_3','Decoder-Stage-1-DepthConcat/in2');
lgraph = connectLayers(lgraph,'relu3_3','Decoder-Stage-2-DepthConcat/in2');
lgraph = connectLayers(lgraph,'relu2_2','Decoder-Stage-3-DepthConcat/in2');
lgraph = connectLayers(lgraph,'relu1_2','Decoder-Stage-4-DepthConcat/in2');
figure, plot(lgraph)