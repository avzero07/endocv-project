%% UNet-Residual-AV Architecture

% @author   - Akshay Viswakumar
% @email    - akshay.viswakumar@gmail.com
% @version  - v1.0

%% Details

% This script is used to illustrate how UNet-Residual-AV was constructed from
% scratch. The script also plots a diagram of the constructed network.
%
% NOTE: This script can be run as is without modification as long as the
% active path in MATLAB is set as the root directory of the Demo package
% (directory containing this script). If not, please set the path or add
% the project folder and its sub-folders to your MATLAB environment's
% active path.

%% Initialization
clc
clear variables
close all

baseDir = "Test Dataset/";

imageDirTest = fullfile(baseDir,'256/Input/');
labelDirTest = fullfile(baseDir,'256/Segmentation/');

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

%% 2. Constructing UNet-Residual-AV

% This section assembles UNet-Residual-AV one layer at a time, and makes appropriate
% skip connections per the UNet design Paradigm. The overall design is
% a variation of UNet-AV with residual blocks added to the encoder
% (downsampling) portion of the network. The decoder (upsampling) portion
% of the network is identical to UNet-AV.

numClasses = length(classNames);

layers = [
    imageInputLayer(imageSize,'Name','Input_Layer')
    
    % Encoder Stage 1
    
    % Encoder Stage 1 Convolution #1
    convolution2dLayer(3,64,'Padding','same','Name','Encoder-Stage-1-Conv_1')
    reluLayer('Name','Relu_1-1')
    
    % Encoder Stage 1 Convolution #2
    convolution2dLayer(3,64,'Padding','same','Name','Encoder-Stage-1-Conv_2')
    additionLayer(2,'Name','Res_Add_1')
    reluLayer('Name','Relu_1-2')
    
    % Encoder Stage 1 Pooling
    maxPooling2dLayer([2 2],'Stride',2,'Name','Encoder-Stage-1-MaxPooling')
    
    % Encoder Stage 2
    
    % Encoder Stage 2 Convolution #1
    convolution2dLayer(3,128,'Padding','same','Name','Encoder-Stage-2-Conv_1')
    reluLayer('Name','Relu_2-1')
    
    % Encoder Stage 2 Convolution #2
    convolution2dLayer(3,128,'Padding','same','Name','Encoder-Stage-2-Conv_2')
    additionLayer(2,'Name','Res_Add_2')
    reluLayer('Name','Relu_2-2')
    
    % Encoder Stage 2 Pooling
    maxPooling2dLayer([2 2],'Stride',2,'Name','Encoder-Stage-2-MaxPooling')
    
    % Encoder Stage 3
    
    % Encoder Stage 3 Convolution #1
    convolution2dLayer(3,256,'Padding','same','Name','Encoder-Stage-3-Conv_1')
    reluLayer('Name','Relu_3-1')
    
    % Encoder Stage 3 Convolution #2
    convolution2dLayer(3,256,'Padding','same','Name','Encoder-Stage-3-Conv_2')
    additionLayer(2,'Name','Res_Add_3')
    reluLayer('Name','Relu_3-2')
    
    % Encoder Stage 3 Pooling
    maxPooling2dLayer([2 2],'Stride',2,'Name','Encoder-Stage-3-MaxPooling')
        
    % Encoder Stage 4
    
    % Encoder Stage 4 Convolution #1
    convolution2dLayer(3,512,'Padding','same','Name','Encoder-Stage-4-Conv_1')
    reluLayer('Name','Relu_4-1')
    
    % Encoder Stage 4 Convolution #2
    convolution2dLayer(3,512,'Padding','same','Name','Encoder-Stage-4-Conv_2')
    additionLayer(2,'Name','Res_Add_4')
    reluLayer('Name','Relu_4-2')
    
    % Encoder Stage 4 Pooling
    maxPooling2dLayer([2 2],'Stride',2,'Name','Encoder-Stage-4-MaxPooling')
    
    % Bridge Stage
    
    % Bridge Stage Convolution
    convolution2dLayer(3,1024,'Padding','same','Name','Bridge-Stage-Conv')
    reluLayer('Name','Relu_Bridge-1')
    
    % Bridge Stage UpConvolution
    transposedConv2dLayer(2,512,'Stride',2,'Cropping','same','Name','Bridge-Stage-Trans_Conv')
    
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
    transposedConv2dLayer(2,256,'Stride',2,'Cropping','same','Name','Decoder-Stage-1-Trans_Conv')
    
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
    transposedConv2dLayer(2,128,'Stride',2,'Cropping','same','Name','Decoder-Stage-2-Trans_Conv')
    
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
    transposedConv2dLayer(2,64,'Stride',2,'Cropping','same','Name','Decoder-Stage-3-Trans_Conv')
    
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

lgraph = layerGraph(layers);

% Skip Connections
lgraph = connectLayers(lgraph,'Relu_4-2','Decoder-Stage-1-DepthConcat/in2');
lgraph = connectLayers(lgraph,'Relu_3-2','Decoder-Stage-2-DepthConcat/in2');
lgraph = connectLayers(lgraph,'Relu_2-2','Decoder-Stage-3-DepthConcat/in2');
lgraph = connectLayers(lgraph,'Relu_1-2','Decoder-Stage-4-DepthConcat/in2');

% Construct and Connect Residual Layers

% Encoder Stage 1 - Residual Block

layersResidual1 = [
        convolution2dLayer(1,64,'Padding','same','Name','Res_1_Conv')
    ];

lgraph = addLayers(lgraph,layersResidual1);
lgraph = connectLayers(lgraph,'Input_Layer','Res_1_Conv');
lgraph = connectLayers(lgraph,'Res_1_Conv','Res_Add_1/in2');

% Encoder Stage 2 - Residual Block

layersResidual2 = [
        convolution2dLayer(1,128,'Padding','same','Name','Res_2_Conv')
    ];

lgraph = addLayers(lgraph,layersResidual2);
lgraph = connectLayers(lgraph,'Encoder-Stage-1-MaxPooling','Res_2_Conv');
lgraph = connectLayers(lgraph,'Res_2_Conv','Res_Add_2/in2');

% Encoder Stage 3 - Residual Block

layersResidual3 = [
        convolution2dLayer(1,256,'Padding','same','Name','Res_3_Conv')
    ];

lgraph = addLayers(lgraph,layersResidual3);
lgraph = connectLayers(lgraph,'Encoder-Stage-2-MaxPooling','Res_3_Conv');
lgraph = connectLayers(lgraph,'Res_3_Conv','Res_Add_3/in2');

% Encoder Stage 4 - Residual Block

layersResidual4 = [
        convolution2dLayer(1,512,'Padding','same','Name','Res_4_Conv')
    ];

lgraph = addLayers(lgraph,layersResidual4);
lgraph = connectLayers(lgraph,'Encoder-Stage-3-MaxPooling','Res_4_Conv');
lgraph = connectLayers(lgraph,'Res_4_Conv','Res_Add_4/in2');
figure, plot(lgraph), title('UNet-Residual-AV')

%% 3. Sample Code for Training on Test Dataset

% NOTE: This section is for purposes of illustration. The Demo package only 
% contains the Test set and this has been used below as both the training 
% and validation sets in absence of actual training and validatio data.
%
% The Lines below (line 267 and below) may be uncommented to run a few iterations of training. 
% The max Epochs is set to 1. Please change this to something else if you 
% want to run the code for a larger number of Epochs.
% 
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

