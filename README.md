# FLBGDNet
This repo contains the KERAS implementation of "Fast Lightweight Blind Gaussian Denoiser using Feature Denoising based Deep Residual Network(FLBGDNet)"


Run Experiments

To test for blind Gray denoising using FLBGDNet write:

python Test_Gray.py

The resultant images will be stored in 'Test_Results/Gray/'

To test for blind Color denoising using FLBGDNet write:

python Test_Color.py

The resultant images will be stored in 'Test_Results/Color/'

Image wise PSNR & SSIM as well as Average PSNR & Average SSIM for the whole image database is also displayed in the console as output.
