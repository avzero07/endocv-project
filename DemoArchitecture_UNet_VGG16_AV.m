%% UNet-VGG16-AV Architecture

% @author   - Akshay Viswakumar
% @email    - akshay.viswakumar@gmail.com
% @version  - v1.0

%% Details

% This script is used to illustrate how UNet-VGG16-AV was constructed. The 
% script also plots a diagram of the constructed network.
%
% NOTE: This script can be run as is without modification as long as the
% active path in MATLAB is set as the root directory of the Demo package
% (directory containing this script). If not, please set the path or add
% the project folder and its sub-folders to your MATLAB environment's
% active path.
%
% NOTE: Since this network uses a portion of a pre-trained VGG16 network,
% it will require the additional MATLAB package (VGG16) to be installed. If
% this is absent from your system, you will be prompted to install this the
% first time this script is run.

%% Initialization
clc
clear variables
close all

baseDir = "Test Dataset/";

imageDirTest = fullfile(baseDir,'224/Input/');
labelDirTest = fullfile(baseDir,'224/Segmentation/');

classNames = ["BE","Suspicious","HGD","Cancer","Polyp","Background"];
labelIDs   = [51 101 151 201 251 0];

imageSize = [256 256 3]; % Can be Modified

%% Implementation

%% 1. Preparing Datastores

% This section is for assigning weights for the pixelClassification layer
% (final layer of the network). The weights are assigned in a way that
% counters class imbalance.

% Load Test Dataset
imdsTest = imageDatastore(imageDirTest);
pxdsTest = pixelLabelDatastore(labelDirTest,classNames,labelIDs);

tbl = countEachLabel(pxdsTest);
frequency = tbl.PixelCount/sum(tbl.PixelCount);

% Balancing Weights
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;

% NOTE: The Test Dataset is loaded here for purposes of illustration. In
% the actual implementation (including the script which was used for
% training), the Training Set is used to solve class imbalance and create a
% pixel classification layer with balanced weights. 

% Here, the Test dataset is used for balancing since the Training set is 
% not included with the Demo package.

%% 2. Constructing UNet-VGG16-AV

% This section assembles UNet-VGG16-AV one layer at a time, and makes appropriate
% skip connections per the UNet design Paradigm. Unlike the other designs,
% this design makes use of a pre-trained VGG16 network (pre-trained on ImageNet) 
% as a backbone which replaces the encoder (downsampling) portion of UNet-AV.
%
% For this, the first 25 layers of a pre-trained VGG16 network is used as
% the encoder network. Following this, a decoder network similar to the one
% in UNet-AV is constructed. Both networks are attached and appropriate
% skip connections are made as per the UNet design paradigm.
%
% Visually this network might look like UNet-AV but the encoder network is
% VGG16 with weights pre-trained on ImageNet. Further training was carried
% out on the EndoCV2020 dataset as part of the project.

numClasses = length(classNames);

% Load Pre-trained VGG16 and Extract Convolutional Layers
VGGnet = vgg16;
lgraphVGG = layerGraph(VGGnet.Layers(1:25)); % Extract Upto Start of Dense

% Construct Additional Layers of the UNet to Assemble UNet-VGG16-AV
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

figure, plot(lgraph), title('UNet-VGG16-AV')

%% 3. Sample Code for Training on Test Dataset

% NOTE: This section is for purposes of illustration. The Demo package only 
% contains the Test set and this has been used below as both the training 
% and validation sets in absence of actual training and validatio data.
%
% The Lines below (line 188 and below) may be uncommented to run a few iterations of training. 
% The max Epochs is set to 1. Please change this to something else if you 
% want to run the code for a larger number of Epochs.

% % Data Augmentation Code
% augmenter = imageDataAugmenter('RandXReflection',true,'RandYReflection',true,'RandRotation',[0 360]);
% 
% ds = pixelLabelImageDatastore(imdsTest,pxdsTest,'DataAugmentation',augmenter);
% dsVal = pixelLabelImageDatastore(imdsTest,pxdsTest);
% 
% %Training options.
% q4options = trainingOptions('adam','InitialLearnRate',3e-6, 'SquaredGradientDecayFactor', 0.99,'ValidationPatience',7,...
%     'MiniBatchSize',10,'MaxEpochs',1,'VerboseFrequency',10,'Shuffle','every-epoch', ...
%     'ValidationData',dsVal, ...
%     'ValidationFrequency',30, ...
%     'Verbose',true, ...
%     'Plots','training-progress','CheckpointPath','Checkpoints\');
% 
% %gpu1 = gpuDevice(1);  % Uncomment If Using a GPU
% 
% % Train
% myUnet = trainNetwork(ds,lgraph,q4options);

