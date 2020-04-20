%% Demo Segmentation

% @author   - Akshay Viswakumar
% @email    - akshay.viswakumar@gmail.com
% @version  - v1.0

%% Details

% This script showcases the Segmentation performances of trained 
% UNet-Residual-AV and UNet-VGG16-AV. Section 5 may be re-run to check
% segmentation outputs for different samples of the Test Dataset.

% NOTE: This script can be run as is without modification as long as the
% active path in MATLAB is set as the root directory of the Demo package
% (directory containing this script). If not, please set the path or add
% the project folder and its sub-folders to your MATLAB environment's
% active path.

%% 1. Implementation

% This section initializes paths. Note that there are two version of the
% test dataset included with the demo package. These are identical except
% the fact that one set has images and masks with dimensions 256x256 and
% while the other is 224x224. 

% This is because UNet-VGG16-AV which uses a VGG16 for it's encoder network
% is built to process images of dimension 224x224.

% Initialization
clc;
clear variables;
close all;

baseDir = "Test Dataset/";

% Load Test Dataset for UNet-Residual-AV
imageDirTest = fullfile(baseDir,'256/Input');
labelDirTest = fullfile(baseDir,'256/Segmentation');

% Load Test Dataset for UNet-VGG16-AV
imageDirTest2 = fullfile(baseDir,'224/Input');
labelDirTest2 = fullfile(baseDir,'224/Segmentation');



%% 2. Create Datastores

% This section defines datastores to be passed to each network.

classNames = ["BE","Suspicious","HGD","Cancer","Polyp","Background"];
labelIDs   = [51 101 151 201 251 0];

% Datastore for UNet-Residual-AV
imdsTest = imageDatastore(imageDirTest);
pxdsTest = pixelLabelDatastore(labelDirTest,classNames,labelIDs);

% Datastore for UNet-VGG16-AV
imdsTest2 = imageDatastore(imageDirTest2);
pxdsTest2 = pixelLabelDatastore(labelDirTest2,classNames,labelIDs);

%% 3. Load Trained Network

% This section loads the pre-trained networks from the TrainedWeights
% Directory.
%
% NOTE: Unlike other frameworks, MATLAB saves a snapshot of the entire network when
% making checkpoints during training. As a result, these networks need not
% be re-constructed during time of evaluation. For details on how these
% networks were constructed, please refer to the appropriate "DemoArchitecture"
% scripts in the Demo package.

% Load Trained UNet-Residual-AV
net1Struct = load('Trained Weights\final-weights-UNet-Residual-AV.mat');
net1 = net1Struct.net;

% Load Trained UNet-VGG16-AV
net2struct = load('Trained Weights\final-weights-UNet-VGG16-AV.mat');
net2 = net2struct.net;

%% 4. Evaluate Metrics over Test Set

% This section evaluates each network over the complete Test Dataset (67)
% images and computes corresponding metrics.

pxdsResultsResidual = semanticseg(imdsTest,net1,'MiniBatchSize',4,'WriteLocation','Output/UNet-Residual-AV/','Verbose',true);
metricsResidual = evaluateSemanticSegmentation(pxdsResultsResidual,pxdsTest,'Verbose',true);

pxdsResultsVGG16 = semanticseg(imdsTest2,net2,'MiniBatchSize',4,'WriteLocation','Output/UNet-VGG16-AV/','Verbose',true);
metricsVGG16 = evaluateSemanticSegmentation(pxdsResultsVGG16,pxdsTest2,'Verbose',true);

% The following Metrics are available
% 1. metricsResidual.ClassMetrics | metricsVGG16.ClassMetrics 
% 2. metricsResidual.DataSetMetrics | metricsVGG16.ClassMetrics
% 3. metricsResidual.NormalizedConfusionMatrix | metricsVGG16.NormalizedConfusionMatrix

% These can be inspected after running this section by calling the
% appropriate variable.
%
% Example: To see the Normalized Confusion Matrix for UNet-VGG16-AV, 
% run 'metricsVGG16.NormalizedConfusionMatrix' in the Command Window.

%% 5. Evaluate for Specific Samples

% This section of the code serves to illustrate the semantic segmentation
% capability of both of the trained networks. The ground truth result is
% also displayed for comparison.

num = 15; 
% Change Value of 'num' above any number between from 1 - 67 and rerun this 
% section alone to sample a different image in the test dataset.
%
% Note: Try 36, 6, 15, 11, 51 for Samples Showcased in the Report

fade = 0.4;

tI = readimage(imdsTest,num);
tS1 = semanticseg(tI, net1);

tIC = readimage(imdsTest2,num);
tS2 = semanticseg(tIC, net2);

tG = readimage(pxdsTest,num);

cmap = endocvCmap();

figure,
A = labeloverlay(tI,tG,'Colormap',cmap,'Transparency',fade);
subplot(1,3,1), imshow(A)
pixelLabelColorbar(cmap,classNames);
title('Ground Truth Segmentation')

B = labeloverlay(tI,tS1,'Colormap',cmap,'Transparency',fade);
subplot(1,3,2), imshow(B)
pixelLabelColorbar(cmap,classNames);
title('UNet-Residual-AV - Predicted')

C = labeloverlay(tIC,tS2,'Colormap',cmap,'Transparency',fade);
subplot(1,3,3), imshow(C)
pixelLabelColorbar(cmap,classNames);
title('UNet-VGG16-AV - Predicted')

sgtitle('Segmentation on Test Samples');