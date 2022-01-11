#importLibraries
from tensorflow.keras.models import load_model
from conf import myConfig as config
import cv2
import numpy as np
import argparse
from pathlib import Path
import tensorflow.keras.backend as K
import tensorflow as tf

#ParsingArguments
parser=argparse.ArgumentParser()
parser.add_argument('--dataPath',dest='dataPath',type=str,default='./Testing_data/Synthetic',help='testDataPath')
parser.add_argument('--weightsPath',dest='weightsPath',type=str,default='./Pretrained_models/SIFSDNet_SAR.h5',help='pathOfTrainedCNN')
args=parser.parse_args()
#createModel, loadWeights
def custom_loss(y_true,y_pred): #this is required for loading a keras-model created with custom-loss
    diff=K.abs(y_true-y_pred)
    res=(diff)/(config.batch_size)
    return res

nmodel_PROPOSED=load_model(args.weightsPath,custom_objects={'custom_loss':custom_loss})
print('Trained Model is loaded')

def ENL(img):
    mean=np.average(img)
    std=np.std(img)
    ENL1=(mean*mean)/(std*std)
    return ENL1

#createArrayOfTestImages
p=Path(args.dataPath)
listPaths=list(p.glob('./*.bmp'))
imgTestArray = []
for path in listPaths:
    imgTestArray.append((
    (cv2.imread(str(path),0))))
imgTestArray=np.array(imgTestArray)/255.

lenth=1
sumPSNR=0
sumSSIM=0
psnr_val=np.empty(lenth)
ssim_val=np.empty(lenth)
for i in range(0,lenth):
    np.random.seed(seed=0)  # for reproducibility
    img1=imgTestArray[i]
    z=np.squeeze(nmodel_PROPOSED.predict(np.expand_dims(img1,axis=0)))
    cv2.imwrite("./Test_Results/SAR/"+str(i+1)+"_Original.png",255.*img1)
    cv2.imwrite("./Test_Results/SAR/"+str(i+1)+"_SIFSDNet.png",255.*z)
    enl=ENL(z)
    print('ENL of image '+str(i+1)+' is ',enl)
