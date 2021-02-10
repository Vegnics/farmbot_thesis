from time import time,sleep
from farmware_tools import get_config_value,device
import cv2
from cv2 import floodFill
import numpy as np
from numpy.fft import fft2,fftshift,ifft2,ifftshift
import os

def usb_camera_photo():
    # 'Take a photo using a USB camera.'#
    camera_port = 0  # default USB camera port
    max_port_num = 1  # highest port to try if not detected on port
    discard_frames = 10  # number of frames to discard for auto-adjust
    max_attempts = 5  # number of failed discard frames before quit
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 640
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 480
    cam.set(cv2.CAP_PROP_BRIGHTNESS, 0.35)  # 0.5
    cam.set(cv2.CAP_PROP_CONTRAST, 0.73)  # 0.733333
    cam.set(cv2.CAP_PROP_SATURATION, 0.5)  # 0.3543
    cam.set(cv2.CAP_PROP_HUE, 0.5)  # 0.5
    device.log(message='setting ok', message_type='success')
    failed_attempts = 0
    max_attempts = 5

    for a in range(10):
        ret, image = cam.read()
        if not cam.grab():
            # verbose_log('Could not get frame.')
            failed_attempts += 1
        if failed_attempts >= max_attempts:
            break
        sleep(0.1)
    # Take a photo
    ret, image = cam.read()
    directory = '/tmp/images/'
    image_filename = directory + '{timestamp}.jpg'.format(timestamp=int(time()))
    cv2.imwrite(image_filename, image)
    # Close the camera
    cam.release()
    return image

def rgb_to_hsv_float(img,add_sat,add_value):
    img=np.clip(img,0.0,255.0)
    img_b = img/255.0
    img_hsv = np.zeros(img_b.shape)
    r=img_b[:,:,2]
    g = img_b[:, :, 1]
    b = img_b[:, :, 0]
    cmax = np.amax(img_b,axis=2) # this is the cmax matrix
    cmin = np.amin(img_b,axis=2) # this is the cmin matrix
    diff = cmax - cmin         # this is the diff matrix
    V = cmax # this is the Value matrix
    cmax_m = np.where(cmax>0,cmax,10)
    S = np.where((V>0.0001)&(cmax_m != 10), diff/cmax_m, 0) #this is the Saturation matrix
    diff = np.where(diff == 0, 10, diff)
    H_b = np.where((np.abs(cmax - b)<0.00000001) & (diff != 10),60*(4+(r-g)/diff),0)
    H_g = np.where((np.abs(cmax - g)<0.00000001) & (diff != 10),60*(2+(b-r)/diff),0)
    H_r = np.where((np.abs(cmax - r)<0.00000001) & (diff != 10),60*(((g-b)/diff)%6),0)
    H=H_b+H_g+H_r
    H = np.where(H<0,H+360,H)
    img_hsv[:,:,0] = H/2
    img_hsv[:,:,1] = S*255 + add_sat
    img_hsv[:,:,2] = V*255 + add_value
    img_hsv = img_hsv.astype(np.uint8)
    img_hsv[:, :, 1] = cv2.equalizeHist(img_hsv[:, :, 1])
    img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:,:,2])
    return img_hsv

def homomorph_filter_N1(src,kernel):
    src = src.astype(np.float32)
    Ln_I = np.log(src + 1)
    I_fft = fft2(Ln_I)
    I_fft = fftshift(I_fft)
    I_filt_fft = I_fft * kernel
    I_filt_fft_uns = ifftshift(I_filt_fft)
    I_filtered = np.real(ifft2(I_filt_fft_uns))
    I_filtered = np.exp(I_filtered) - 1
    return I_filtered

def homomorph_filter_N3(src,kernel):
    outimg = np.zeros(src.shape)
    B, G, R = cv2.split(src)
    nB = homomorph_filter_N1(B, kernel)
    nG = homomorph_filter_N1(G, kernel)
    nR = homomorph_filter_N1(R, kernel)
    outimg[:, :, 0] = nB
    outimg[:, :, 1] = nG
    outimg[:, :, 2] = nR
    return outimg

dir_path = os.path.dirname(os.path.realpath(__file__))
butt_kernel = np.load(dir_path+'/'+'kernel_butt.npy')

img=usb_camera_photo()
I_filtered = homomorph_filter_N3(img,butt_kernel)
I_hsv = rgb_to_hsv_float(I_filtered,0,0)
I_filtered = cv2.cvtColor(I_hsv,cv2.COLOR_HSV2BGR)
directory = '/tmp/images/'
image_filename = directory + '{timestamp}.jpg'.format(timestamp=int(time()))
cv2.imwrite(image_filename, I_filtered)



