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

To train the AGSDNet denoising network, first download the UC Merced Land Use data and copy the images into genData folder. Then generate the training data using:

python generateData.py

This will save the training patch 'img_clean_pats.npy' in the folder 'trainingPatch/'

Then run the AGSDNet model file using:

python AGSDNet_train.py

This will save the 'AGSDNet.h5' file of in the folder 'Pretrained_models/'.

# AGSDNet architecture image
![MFEBDN_Architecture](https://user-images.githubusercontent.com/89151608/148807575-b518568c-0079-4e6b-ac8b-29c01972e40e.png)

# AGSDNet synthetic image denoising comparison
![MFEBDN_Gray_Denoising_Comparison](https://user-images.githubusercontent.com/89151608/148807795-c09c5b4c-5476-4f08-b27e-2c40d95bf035.png)
