# MFEBDN
This repo contains the KERAS implementation of "Blind Gaussian Deep Denoiser Network using Multi-Scale Feature Extraction(MFEBDN)"


Run Experiments

To test for blind Gray denoising using MFEBDN write:

python Test_Gray.py

The resultant images will be stored in 'Test_Results/Gray/'

To test for blind Color denoising using MFEBDN write:

python Test_Color.py

The resultant images will be stored in 'Test_Results/Color/'

Image wise PSNR & SSIM as well as Average PSNR & Average SSIM for the whole image database is also displayed in the console as output.
