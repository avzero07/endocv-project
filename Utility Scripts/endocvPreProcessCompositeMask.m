%% Pre Processing - Composite Masks
% @author   - Akshay Viswakumar
% @email    - akshay.viswakumar@gmail.com
% @version  - v1
% @date     - 20-March-2020

%% Details

% This script is used to create composite masks using the original masks
% provided in the EndoCV2020 dataset.

% NOTE: The test dataset accompanying the demo have already been
% pre-processed. There is no need to run this script for the purposes of
% the demo. This has been included within the package since this was one of
% the utility functions created as part of this project for pre-processing
% masks from the original dataset.

% This script can be re-used to generate composite masks for data from the
% EndoCV2020 dataset or similar datasets. Please make sure to modify the
% following parts of the script to suit your dataset.
%
% 1. Replace All Paths in the 'Initialization' section with paths suitable
%    to your system environment.
%
% 2. Update the "opSize" variable in the 'Initialization' section with the
%    image size of your specific dataset.

%% Implementation
clc;
clear variables
close all

%% Initialization

% Modify all paths according to system environment.
basePath = "C:\Users\aksha\Documents\Projects\endocv-project\Dataset\";

pathToImages = basePath+"image-resized\"; 
pathToMasks = basePath+"mask-resized\";
pathOP = basePath+"mask-resized-combined\";

opSize = [256 256]; % Modify According to Dataset being Used
%% Get List of Filenames

imageStruct = dir(pathToImages);
imageFileName = {imageStruct.name}; % List of Image Filenames

maskStruct = dir(pathToMasks);
maskFileName = {maskStruct.name}; % List of Mask Filenames

%% Loop Through Masks and Create a Composite Mask in the Output Directory

% This portion of the script, loops through

for i=3:length(imageFileName)
    fileName = cell2mat(imageFileName(i)); % Filename
    opMask = zeros(opSize); % OP Mask Image
    
    fileName
    for j=3:length(maskFileName)
        maskFName = cell2mat(maskFileName(j)); % Mask Filename
        
        k = strfind(maskFName,'_'); % Get Index of Second _
        class = maskFName(k(2)+1:end-4); % Get Class Name
        
        fillVal = 0;
        % Set Fill Value
        if(strcmp(class,'BE'))
            fillVal = 51;
        elseif(strcmp(class,'suspicious'))
            fillVal = 101;
        elseif(strcmp(class,'HGD'))
            fillVal = 151;
        elseif(strcmp(class,'cancer'))
            fillVal = 201;
        elseif(strcmp(class,'polyp'))
            fillVal = 251;
        end
        
        if(contains(maskFName,fileName(1:end-4)))
            maskFName
            mask = imread(pathToMasks+maskFName);
            mask(mask>0) = fillVal;
            opMask(mask==fillVal) = fillVal;
        end
    end
    imwrite(uint8(opMask),pathOP+fileName,'png');
end