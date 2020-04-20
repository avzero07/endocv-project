# endocv-project
Semantic Segmentation of Enoscopic Images Using UNet-like Networks

This Demo package contains some scripts used to showcase the results of this project.

--------------------
System Requirements
--------------------

1. Hardware Requirements

- Any modern computer system (2014 or later) should be sufficient. 
- A GPU is not necessary since training will not be performed.

2. Software Requirements

- All development was carried out on MATLAB r2019b. All higher versions of MATLAB should be able to run all code without issue. Lower versions of MATLAB upto r2017a may be compatible since all associated functions and toolboxes are available starting version r2017. That said, this Demo package has not been tested on lower versions of MATLAB and so I cannot guarantee the performance.

- Please make sure the following MATLAB toolboxes and packages are present.
	- Deep Learning Toolbox
	- Parallel Computing Toolbox
	- Computer Vision and Image Processing Toolbox
	- VGG16 Package (for running 'DemoArchitecture_UNet_VGG16_AV.m')

----------------------------
Description of Demo Package
----------------------------

The purpose of the Demo package is to showcase the work carried out as part of the EECE-570 course project titled "Semantic Segmenation of Endoscopic Images". The following 4 scripts are included as part of the Demo.

1. 'DemoArchitecture_UNet_AV.m'		 - This script illustrates how UNet-AV was constructed from scratch. 
2. 'DemoArchitecture_UNet_Residual_AV.m' - This script illustrated how UNet-Residual-AV was constructed from scratch.
3. 'DemoArchitecture_UNet_VGG16_AV.m'	 - This script illustrated how UNet-VGG16-AV was constructed from scratch.
4. 'DemoSegmentation.m'			 - This script demonstrates the segmentation capability of trained versions of UNet-Residual-AV and UNet-VGG16-AV.

Please run these scripts. Refer to individual scripts for more comments and details.

IMPORTANT: These scripts have been tested and can be run without modification as long the active path in MATLAB is set as the root directory of the Demo package (directory containing this script). If not, please set the path or add the project folder and its sub-folders to your MATLAB environment's active path.

----------------------------
Other Contents of Demo Package
----------------------------

1. Trained versions of UNet-Residual-AV and UNet-VGG16-AV. These are versions of the networks trained on the Training portion of the EndoCV dataset. Reported results are from these version of the networks. The trained networks are located in the 'Trained Weights' directory.

2. The 'Test Dataset' directory contains two versions of the Test Dataset each comprising 67 images and their corresponding composite masks. These are identical except the fact that one set has images and masks with dimensions 256x256 while the other is with dimensions 224x224. This is because UNet-VGG16-AV which uses a VGG16 for it's encoder network is built to process images of dimension 224x224

3. 'endocvCmap.m' and 'pixelLabelColorbar.m' are helper functions used in 'DemoSegmentation.m'. Please do not delete these files.

4. The 'Utility Scripts' folder contains some scripts used for pre-processing the images from the original EndoCV dataset. These are not designed to be run as part of the Demo package. They are merely included here for completeness. A pre-processed version of the Test Data is included with the Demo package. That said, these scripts may be modified for pre-processing images from the original EndoCV2020 dataset or similar datasets. Refer to the comments in each script for more info. Note that the scripts are designed to be run in the following order.
	a. 'endocvPreProcessResize.m'		- Resizes all images and masks to a common dimension which the user can specify.

	b. 'endocvPreProcessCompositMask.m'	- After resizing, masks are processed to create composite masks with one mask corresponding to each image. For more information on why this is necessary, please refer to the Section title 'Dataset' of the accompanying report.

	c. 'endocvPreProcessSplitData.m'	- Splits the processed dataset into Training, Validation and Test sets.

5. 'Checkpoints' and 'Output' folders are made use of in the Demo scripts. These may be ignored.
