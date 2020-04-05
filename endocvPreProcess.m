%%
clc;
clear variables
close all

%% Init
valPath = "C:\Users\aksha\Documents\Github Projects\endocv-project\Dataset\Seg-Val\";
trainPath = "C:\Users\aksha\Documents\Github Projects\endocv-project\Dataset\Seg-Train\";
testPath = "C:\Users\aksha\Documents\Github Projects\endocv-project\Dataset\Seg-Test\";

pathToImages = "C:\Users\aksha\Documents\Github Projects\endocv-project\Dataset\originalImages-resized-2\";
pathToMasks = "C:\Users\aksha\Documents\Github Projects\endocv-project\Dataset\masks-resized-2\";

pathOP = "C:\Users\aksha\Documents\Github Projects\endocv-project\Dataset\masks-resized-combined\";

%% Get List of Filenames
imageStruct = dir(pathToImages);
imageFileName = {imageStruct.name};

maskStruct = dir(pathToMasks);
maskFileName = {maskStruct.name};

%% Loop Through Files

for i=3:length(imageFileName)
    fileName = cell2mat(imageFileName(i)); % Filename
    opMask = zeros(256,256); % OP Mask Image
    
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
    imwrite(uint8(opMask),pathOP+fileName,'jpg');
end

%% Split Data into Train, Test and Validate

rng(0,'twister'); % For Repeatability

r = randi([1 386],1,77);

% Split Data into Train / Test
for i=3:length(imageFileName)
    fileName = cell2mat(imageFileName(i));
    if(ismember(i-2,r))
        i
        copyfile(pathToImages+fileName,testPath+'Input\');
        copyfile(pathOP+fileName,testPath+'Segmentation\')
    else
        copyfile(pathToImages+fileName,trainPath+'Input\');
        copyfile(pathOP+fileName,trainPath+'Segmentation\')
    end
    %fileName
end

%% Split Data into Train / Validate
trainFileNameStruct = dir(trainPath+'Input\');
trainFileName = {trainFileNameStruct.name};

rng(0,'twister'); % For Repeatability
r = randi([1 386],1,77);
r2 = randi([1 309],1,62);

for i=3:length(trainFileName)
    fileName = cell2mat(trainFileName(i));
    
    if(ismember(i-2,r2))
        copyfile(trainPath+'Input\'+fileName,valPath+'Input\');
        delete(trainPath+'Input\'+fileName)
        copyfile(trainPath+'Segmentation\'+fileName,valPath+'Segmentation\');
        delete(trainPath+'Segmentation\'+fileName)
    end
    fileName
end