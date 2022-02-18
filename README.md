# AGSDNet
This repo contains the KERAS implementation of "AGSDNet: Attention and Gradient based SAR Denoising Network"

# Run Experiments

To test for SAR denoising using AGSDNet write:

python Test_SAR.py

The resultant images will be stored in 'Test_Results/SAR/'

Image wise ENL for the whole image database will also be displayed in the console as output.

To test for synthetic denoising using AGSDNet write:

python Test_Synthetic.py

The resultant images will be stored in 'Test_Results/Synthetic/'

Image wise PSNR & SSIM as well as Average PSNR & Average SSIM for the whole image database will also be displayed in the console as output.

# Train AGSDNet denoising network

To train the AGSDNet denoising network, first download the [UC Merced Land Use data](http://weegee.vision.ucmerced.edu/datasets/landuse.html) and copy the images into genData folder. Then generate the training data using:

python generateData.py

This will save the training patch 'img_clean_pats.npy' in the folder 'trainingPatch/'

Then run the AGSDNet model file using:

python AGSDNet_train.py

This will save the 'AGSDNet.h5' file of in the folder 'Pretrained_models/'.

# AGSDNet architecture image
![image](https://user-images.githubusercontent.com/89151608/154666534-a00d3de9-37d3-4bd6-9525-3ee6dc9948e4.png)

# AGSDNet synthetic image denoising comparison
![image](https://user-images.githubusercontent.com/89151608/154666800-117660f2-25b8-4ec2-abb9-7eae98f8fbde.png)

# AGSDNet SAR image denoising comparison
![image](https://user-images.githubusercontent.com/89151608/154667006-01245652-5db0-410f-bc5d-2af9f208c866.png)
