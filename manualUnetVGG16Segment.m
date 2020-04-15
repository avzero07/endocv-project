%% Manual UNetVGG16 Segmentation
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

baseDir = "Dataset\Custom-Size\";

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
labelIDs   = [51 101 151 201 251 0];
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

% Balancing Weights
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);

% Data Augmentation Code
augmenter = imageDataAugmenter('RandXReflection',true,'RandYReflection',true,'RandRotation',[0 360]);

%% Manually Create U-Net_VGG16

VGGnet = vgg16;
lgraphVGG = layerGraph(VGGnet.Layers(1:25)); % Extract Upto Start of Dense

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
lgraph = connectLayers(lgraph,'pool4','Bridge-Stage-Conv');

% Skip Connections
lgraph = connectLayers(lgraph,'relu4_3','Decoder-Stage-1-DepthConcat/in2');
lgraph = connectLayers(lgraph,'relu3_3','Decoder-Stage-2-DepthConcat/in2');
lgraph = connectLayers(lgraph,'relu2_2','Decoder-Stage-3-DepthConcat/in2');
lgraph = connectLayers(lgraph,'relu1_2','Decoder-Stage-4-DepthConcat/in2');
figure, plot(lgraph)

%% Transform Data
targetSize = [224 224];

ds = pixelLabelImageDatastore(imds,pxds,'DataAugmentation',augmenter);
dsVal = pixelLabelImageDatastore(imdsVal,pxdsVal);

%% Prepare to Train
%Set up training options.
q4options = trainingOptions('adam','InitialLearnRate',3e-6, 'SquaredGradientDecayFactor', 0.99,'ValidationPatience',7,...
    'MiniBatchSize',10,'MaxEpochs',100,'VerboseFrequency',10,'Shuffle','every-epoch', ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','training-progress','CheckpointPath','Checkpoints\100-Trial-VGG16\');

gpu1 = gpuDevice(1);

%% Train
myUnet = trainNetwork(ds,lgraph,q4options);

%% Evaluate

pxdsResults = semanticseg(imdsTest,myUnetResume,'MiniBatchSize',4,'WriteLocation',outputPath,'Verbose',true);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',true);

%% Test on 1 Image

num = 3;
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
load('Checkpoints\100-Trial-VGG16\net_checkpoint__2600__2020_04_14__17_44_10.mat') % Loads Network as 'net'
%plot(layerGraph(net)) % Plot Loaded Network

%% Resume Training
myUnetResume = trainNetwork(ds,layerGraph(myUnetResume),q4options);