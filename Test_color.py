#importLibraries
from tensorflow.keras.models import load_model
from conf import myConfig as config
import cv2
import numpy as np
import argparse
from pathlib import Path
import tensorflow.keras.backend as K
from utils import utils_image as util


#ParsingArguments
parser=argparse.ArgumentParser()
parser.add_argument('--dataPath',dest='dataPath',type=str,default='./Testing_data/CBSD68',help='testDataPath')
parser.add_argument('--weightsPath',dest='weightsPath',type=str,default='./Pretrained_models/MFEBDN_Color.h5',help='pathOfTrainedCNN')
args=parser.parse_args()
#createModel, loadWeights
def custom_loss(y_true,y_pred): #this is required for loading a keras-model created with custom-loss
    diff=y_true-y_pred
    res=K.sum(diff*diff)/(2*config.batch_size)
    return res

nmodel_PROPOSED=load_model(args.weightsPath,custom_objects={'custom_loss':custom_loss})
print('Trained Model is loaded')

#createArrayOfTestImages
p=Path(args.dataPath)
listPaths=list(p.glob('./*.png'))
imgTestArray = []
for path in listPaths:
    imgTestArray.append((
    (cv2.imread(str(path)))))
imgTestArray=np.array(imgTestArray)/255.

noise_level_img = 25             # noise level for noisy image
lenth=68
sumPSNR=0
sumSSIM=0
psnr_val=np.empty(lenth)
ssim_val=np.empty(lenth)
for i in range(0,lenth):
    np.random.seed(seed=0)  # for reproducibility
    img1=imgTestArray[i]
    f=img1 + np.random.normal(0, noise_level_img/255., img1.shape)
    error=nmodel_PROPOSED.predict(np.expand_dims(f,axis=0))
    predClean=f-np.squeeze(error)
    z=(predClean)
    cv2.imwrite("./Test_Results/Color/"+str(i+1)+"_Original.png",255.*img1)
    cv2.imwrite("./Test_Results/Color/"+str(i+1)+"_Noisy.png",255.*f)
    cv2.imwrite("./Test_Results/Color/"+str(i+1)+"_MFEBDN_Color.png",255.*z)
    psnr_val[i]=util.calculate_psnr(255.*z,255.*img1)
    ssim_val[i]=util.calculate_ssim(255.*z,255.*img1)
    print('PSNR of image '+str(i+1)+' is ',psnr_val[i])
    print('SSIM of image '+str(i+1)+' is ',ssim_val[i])
    sumPSNR=sumPSNR+psnr_val[i]
    sumSSIM=sumSSIM+ssim_val[i]
avgPSNR=sumPSNR/lenth
avgSSIM=sumSSIM/lenth
print('avgPSNR on Set5 dataset = ',avgPSNR)
print('avgSSIM on Set5 dataset = ',avgSSIM)