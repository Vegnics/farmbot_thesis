from time import time,sleep
from farmware_tools import get_config_value,device
import cv2
from cv2 import floodFill
import numpy as np
from numpy.fft import fft,fft2,fftshift,ifft2,ifftshift
import os

def usb_camera_photo():
    # 'Take a photo using a USB camera.'#
    camera_port = 0  # default USB camera port
    max_port_num = 1  # highest port to try if not detected on port
    discard_frames = 10  # number of frames to discard for auto-adjust
    max_attempts = 5  # number of failed discard frames before quit
    cam = cv2.VideoCapture(0)
    start = time()
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 640
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 480
    cam.set(cv2.CAP_PROP_BRIGHTNESS, 0.35)  # 0.5
    cam.set(cv2.CAP_PROP_CONTRAST, 0.73)  # 0.733333
    cam.set(cv2.CAP_PROP_SATURATION, 0.5)  # 0.3543
    cam.set(cv2.CAP_PROP_HUE, 0.5)  # 0.5
    end = time()
    elapsed = end-start
    device.log(message='setting time= {} seconds'.format(elapsed), message_type='success')
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

def enhance_hsv(img):
    img=np.clip(img,0.0,255.0)
    img = np.uint8(img)
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = cv2.equalizeHist(img_hsv[:, :, 1])
    img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:,:,2])
    I_filtered = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
    return I_filtered
    

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

def preprocess(src,kernel):
    src_hsv=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    val = src_hsv[:,:,2]
    I_filtered = homomorph_filter_N3(src,kernel)
    I_filtered = np.clip(I_filtered, 0.0, 255.0)
    I_filtered = np.uint8(I_filtered)
    img_hsv = cv2.cvtColor(I_filtered, cv2.COLOR_BGR2HSV)
    # img_hsv[:, :, 1] = cv2.equalizeHist(img_hsv[:, :, 1])
    #img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    img_hsv[:, :, 2] = val
    I_filtered = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
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

def remove_noise(mask,thresh):
    new_mask=np.zeros(mask.shape)
    num_labels, labeled = cv2.connectedComponents(mask)
    for label in range(num_labels):
        num_pix = np.where(labeled==label)
        if num_pix[0].size>thresh:
            aux_mask = np.where((labeled==label)&(labeled>0),255,0)
            new_mask = new_mask + aux_mask
            new_mask = new_mask.astype(np.uint8)
    return new_mask

def hole_filling(mask,thresh):
    new_mask = np.zeros(mask.shape,dtype=np.uint8)
    mask_inv = cv2.bitwise_not(mask)
    mask_out = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    num_labels, labeled = cv2.connectedComponents(mask_inv)
    for label in range(num_labels):
        coords = zip(*np.where(labeled==label))
        coords = list(coords)
        if len(coords)<thresh and len(coords)>3:
            length = len(coords)
            middle = int(length/2)+1
            mask_out_aux = np.zeros(mask_out.shape, dtype=np.uint8)
            mask_temp = mask.copy()
            flooded = cv2.floodFill(mask_temp, mask_out_aux, (coords[middle][1],coords[middle][0]), 255)
            border = flooded[1]*255
            new_mask = new_mask | mask_temp | np.uint8(border)
    return new_mask

def calc_centroid_distance(contour):
    moments = cv2.moments(contour)
    c_row=moments["m01"]/moments["m00"]
    c_col=moments["m10"]/moments["m00"]
    contour_array = np.array(contour)
    contour_array = np.resize(contour_array,(-1,2))
    d_col = contour_array[:,0] - c_col
    d_row = contour_array[:,1] - c_row
    c_funct = np.sqrt(np.square(d_col) + np.square(d_row))
    return c_funct

def calc_fourier_descriptor(c_function):
    fourier_desc = fft(c_function)
    return fourier_desc

def calc_normalized_fourier(contour):
    c_funct = calc_centroid_distance(contour)
    fourier_desc = calc_fourier_descriptor(c_funct)
    normalized_descriptors = fourier_desc/fourier_desc[1]
    normalized_descriptors = np.abs(normalized_descriptors)
    return normalized_descriptors

def compare_fourier_descriptors(fourier1,fourier2,N=10):
    Fourier1 = fourier1[1:N+1]
    Fourier2 = fourier2[1:N+1]
    error = Fourier1 - Fourier2
    error_sq = np.square(error)
    multipliers = np.linspace(1,100,N)
    poderated_error_sq = error_sq * multipliers
    rms_error = np.sqrt(np.sum(poderated_error_sq)/N)
    return rms_error

def pixel2coord(pixelcoord,actual_pos,z_dist,intrinsics,rmatrix,tvec):
    vx=pixelcoord[1]
    uy=pixelcoord[0]
    
    ppx=intrinsics[0,2]
    ppy=intrinsics[1,2]
    fx=intrinsics[0,0]
    fy=intrinsics[1,1]

    rinvmat=np.linalg.inv(rmatrix)
    tx=tvec[0] #- 5 #it's possible to change the traslation value
    ty=tvec[1] #- 5 #it's possible to change the traslation value
    tz=tvec[2]
    
    X=abs(z_dist)*(vx-ppx)/fx
    Y=abs(z_dist)*(uy-ppy)/fy
    Z=abs(z_dist)
    P=np.array([X,Y,Z])
    t=np.resize(np.array([tx,ty,tz]),(3,1))
    P = np.resize(P, (3, 1))
    P[0] = P[0] - tx
    P[1] = P[1] - ty
    P[2] = P[2] 
    P=np.matmul(rinvmat,P)
    
    P[0] = actual_pos[0] + P[0]
    P[1] = actual_pos[1] + P[1]
    P[2] = actual_pos[2] + P[2]
    #print(rmatrix)
    #print("X= {}, Y={}, Z={}".format(P[0],P[1],P[2])) #
    #print("\n")
    return P

def move_absolute(position,offset,speed):
    as_position= device.assemble_coordinate(position[0],position[1],position[2])
    as_offset=device.assemble_coordinate(offset[0],offset[1],offset[2])
    device.move_absolute(as_position, speed=speed, offset=as_offset)
    
#OBTAINING CONSTANT DATA: HOMO KERNEL, CALIBRATION PARAMETERS, DESCRIPTORS
dir_path = os.path.dirname(os.path.realpath(__file__))
butt_kernel = np.load(dir_path+'/'+'kernel_butt.npy')
descriptors = np.load(dir_path+'/'+'all_descriptors.npy')
tvec = np.load(dir_path+'/'+'tvec.npy')
rmatrix = np.load(dir_path+'/'+'rmatrix.npy')
intrinsics = np.load(dir_path+'/'+'nintrinsics.npy')
matrix=np.load(dir_path+'/'+'array1.npy')
matrix2=np.load(dir_path+'/'+'array2.npy')
matrix3=np.load(dir_path+'/'+'array3.npy')
matrix4=np.load(dir_path+'/'+'array4.npy')

weeder=(23,551,-399)

device.log(message='descriptors shape= {}'.format(descriptors.shape), message_type='success')
#

#MOVEMENTS
#capture_pos = device.assemble_coordinate(500,400,0)
#device.move_absolute(capture_pos, speed=100, offset=device.assemble_coordinate(0,0,0))
move_absolute((500,400,0),(0,0,0),100)

#OBTAINING THE IMAGE
img=usb_camera_photo()
#img = cv2.imread(dir_path+'/'+'image_orig3.jpeg',1)
#img = cv2.imread(dir_path+'/'+'image_orig3_mod.jpeg',1)


#PREPROCESSING
start = time()
#I_filtered = homomorph_filter_N3(img,butt_kernel)
#I_filtered = enhance_hsv(I_filtered)
I_filtered = preprocess(img,butt_kernel)

stop = time()
device.log(message='homofilter_time= {}'.format(stop-start), message_type='success')

directory = '/tmp/images/'
image_filename = directory + '{timestamp}.jpg'.format(timestamp=int(time()))

#SEGMENTATION
H=[20,85]
S=[30,255]
V=[40,255]

kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))

#MORPHOLOGICAL OPERATIONS
start = time()
I_filtered_HSV = cv2.cvtColor(I_filtered,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(I_filtered_HSV,np.array([H[0],S[0],V[0]]),np.array([H[1],S[1],V[1]]))
stop = time()
device.log(message='segmentation_time= {}'.format(stop-start), message_type='success')

start = time()
mask = remove_noise(mask,300)
stop = time()
device.log(message='remove_noise_time= {}'.format(stop-start), message_type='success')

start = time()
mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,kernel_morph,iterations=3)
mask = hole_filling(mask,500)
mask = cv2.morphologyEx(mask,cv2.MORPH_ERODE,kernel_morph,iterations=3)
stop = time()
device.log(message='morphological_time= {}'.format(stop-start), message_type='success')

#CONTOUR EXTRACTION AND ANALYSIS
start = time()
img_segmented= cv2.bitwise_and(I_filtered,I_filtered,mask=mask)
_,contours,hier = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
stop = time()
device.log(message='contour_extraction_time= {}'.format(stop-start), message_type='success')

image_filename = directory + '{timestamp}.jpg'.format(timestamp=int(time()))
cv2.imwrite(image_filename, img_segmented)  

#####move_absolute(weeder,(0,0,15),100)
move_absolute(weeder,(0,0,0),100)
move_absolute(weeder,(100,0,0),100)
move_absolute(weeder,(100,0,100),100)
move_absolute(weeder,(100,0,260),100)

#write_pin(number=4, value=1, mode=0)
#wait(100)
#write_pin(number=4, value=0, mode=0)
#wait(200)
#write_pin(number=4, value=1, mode=0)

#FIND CONTOURS, DETECT SEEDLINGS AND CLASSIFY
seedlings=[]
#seedlings[i]=[[xi,yi],ri,ai] localization,radius,area
for cnt in contours:
    mins = []
    times = []
    if len(cnt)>61:
        start = time()
        descriptor = calc_normalized_fourier(cnt)
        cv2.drawContours(img_segmented,cnt,-1,[0,0,255],3)
        for desc in descriptors:
            start = time()
            D = compare_fourier_descriptors(descriptor, desc, N=60)
            mins.append(D)
            stop = time()
            times.append(stop-start)
            #device.log(message='compare = {}'.format(D), message_type='success')
            if D < 0.8:
                moments = cv2.moments(cnt)
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                (_cx,_cy),r  = cv2.minEnclosingCircle(cnt)
                P = pixel2coord([cy,cx],[500.0,400.0,0.0],485.0 ,intrinsics,rmatrix,tvec)#Change Z
                device.log(message='Matched = {}'.format(D), message_type='success')
                cv2.drawContours(img_segmented,cnt,-1,[0,0,255],3)
                seedlings.append([P[0],P[1],r,moments['m00']])
                #aux=np.abs(P[0]-matrix[:,:,0])+np.abs(P[1]-matrix[:,:,1])
                #(min,_,minloc,_)=cv2.minMaxLoc(aux,None)
                #xmat=minloc[0]-1
                #ymat=minloc[1]-1
                #x,y=matrix[ymat,xmat]
                #move_absolute(x,y,-100),(0,0,0),100)
                break
        min_time = np.min(times)
        max_time = np.max(times)
        mean_time = np.mean(times)
        min = np.min(mins)
        #device.log(message='Found at= {}'.format(P), message_type='success')
        #cv2.putText(img_segmented, "min ={:1.2f}".format(min), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255,0,0],2)
        #move_absolute((int(P[0]),int(P[1]),-100),(0,0,0),100)
        #device.log(message='min_time = {}'.format(min_time), message_type='success')
        #device.log(message='max_time = {}'.format(max_time), message_type='success')
        #device.log(message='mean_time = {}'.format(mean_time), message_type='success')
image_filename = directory + '{timestamp}.jpg'.format(timestamp=int(time()))
cv2.imwrite(image_filename, img_segmented)

device.log(message='Seedling detection OK', message_type='success')

if len(seedling)>0:
    move_absolute(weeder,(0,0,0),100)
    move_absolute(weeder,(100,0,0),100)
    move_absolute(weeder,(100,0,100),100)
    move_absolute(weeder,(100,0,260),100)
    for seedling in seedlings:
        xs,ys=seedling[0]
        aux=np.abs(xs-matrix[:,:,0])+np.abs(ys-matrix[:,:,1])
        (min,_,minloc,_)=cv2.minMaxLoc(aux,None)
        xmat=minloc[0]-1
        ymat=minloc[1]-1
        x,y=matrix[ymat,xmat]
        move_absolute(x,y,-100),(0,0,0),100)
        device.log(message='Seedling found at = {} with r= {} and A= {}'.format(seedling[0],seedling[1],seedling[2]), message_type='success')
    move_absolute(weeder,(120,0,200),100)
    move_absolute(weeder,(120,0,0),100)
    move_absolute(weeder,(0,0,0),100)
    move_absolute(weeder,(0,0,200),100)
    
move_absolute((0,0,0),(0,0,0),100)
device.log(message='Process finished OK', message_type='success')   

