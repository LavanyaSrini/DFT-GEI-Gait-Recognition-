import cv2
import numpy as np
import math
from time import time
import os
from matplotlib import pyplot as plt
import Gait_functions as gf
import pickle
import imutils
import numpy as np
import matplotlib.pyplot as plt
import PIL
import cmath
from PIL import Image


# Pre-process the silhoutte images
print('Enter Input_Path: ')
text = input()

img_path = (text)
Images = sorted(os.listdir(img_path))  
frame_total = len(Images)
GSV = np.zeros((128, 88, frame_total), np.uint8)
frame_num = 0
for img in Images:
	frame = cv2.imread(img_path+img, 0)
	frame = gf.Imcrop_row(frame)
	frame = imutils.resize(frame, height = 128, inter = cv2.INTER_NEAREST)
	frame = gf.CenteringY(frame)
	com = gf.CenterOfMass(frame)
	frame = frame[:, com[1]-44:com[1]+44]
	GSV[:,:,frame_num] = frame
	frame_num+= 1


# Gait period/cycle detection
shift_min = 10
shift_max = 20
shift_tot = len(range(shift_min, shift_max))+1
ac = np.zeros(shift_tot)
for frame_shift in range(shift_tot):
	# Compute the autocorrelation for the GSV for the the frame shift given by frame_shift
	ac[frame_shift] = gf.normalized_correlation(GSV, frame_shift+shift_min, frame_total)
	
gait_period = np.argmax(ac)
gait_period = int(1.5*(gait_period + shift_min))
print("The gait period cycle is %d", gait_period)


def distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2+ (point1[1]-point2[1])**2)


def gaussianLP(D0,imgshape):
    base=np.zeros(imgshape[:2])
    rows,cols=imgshape[:2]
    center=(rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x]=np.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base


def gaussianHP(D0,imgshape):
    base=np.zeros(imgshape[:2])
    rows,cols=imgshape[:2]
    center=(rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x]=1-np.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base


img_G=np.zeros((128,88), np.float64)

for i in range(gait_period+1):
    img_G= img_G+GSV[:,:,i]
img_G=img_G/(gait_period + 1)
original= np.fft.fft2(img_G)
center=np.fft.fftshift(original)
LowPassCenter =center*gaussianLP(50,center.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass=np.fft.ifft2(LowPass)
HighPassCenter =center*gaussianHP(50,center.shape)
HighPass = np.fft.ifftshift(HighPassCenter)
inverse_HighPass=np.fft.ifft2(HighPass)
inverse_LowPass=np.abs(inverse_LowPass)
#inverse_LowPass=Image.fromarray(inverse_LowPass)
inverse_LowPass = np.array(inverse_LowPass)

cv2.imwrite(text+'LP.jpg',inverse_LowPass)

    
#fig =plt.figure(frameon=False)
#fig.set_size_inches(w,h)
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.set_axis_off()
#fig.add_axes(ax)
#ax.imshow(np.abs(inverse_LowPass),'gray')
#fig.savefig(text+'/Gaussian Low Pass.png')


