#!/usr/bin/env python
# coding: utf-8


import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import random


def BRG2RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def readIMG(filename,flagColor = 1):
        if flagColor == 0:
            return cv2.imread(filename,flagColor).astype(np.float32)
        else:
            return BRG2RGB(cv2.imread(filename,flagColor)).astype(np.float32)

def normalizeM(m):
    if len(m.shape) == 3 and m.shape[2] == 3:  # tribanda
        for i in range(3):
            imax, imin = m[:,:,i].max(), m[:,:,i].min()
            if imax == imin:
                m[:,:,i] = 0
            else:
                m[:,:,i] = ((m[:,:,i] - imin)/(imax - imin)) 
    elif len(m.shape) == 2:    # monobanda
        imax, imin = m.max(), m.min()
        if imax == imin:
            m = 0
        else:
            m = ((m - imin)/(imax - imin))
    # escalamos la matriz
    m *= 255
    return m

def pintaI(im,title = "img"):
    
    # Normalize [0,255] as integer
    img = np.copy(normalizeM(im))
    img = np.copy(im)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_GRAY2BGR).astype(np.float64)
        img = img.astype(np.uint8)
        img = BRG2RGB(img)
    else :
        img = img.astype(np.uint8)
   
    
    plt.title(title)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    cv2.waitKey(0)


def convolution2D(img,kx,ky,border):
    """ 
    Convolves img with 2 kernels(one in each axis). Uses different kinds of borders
    """
    # flip kernels and transpose kx
    kx,ky = np.flip(kx).T, np.flip(ky)

    #Apply rows convolution and then, columns convolution
    blurredR = cv2.filter2D(img,-1,kx,  borderType = border)
    blurredC = cv2.filter2D(blurredR,-1,ky, borderType = border)
    return blurredC

def gaussian2D(img,sigma,ksize = 0,border = cv2.BORDER_CONSTANT):
    """ 
    Applies a gaussian filter to a img
    ksize is the dimension
    If kernel size = 0, we calculate it by ksize = 2*3*sigma + 1
    """
    if ksize == 0 :
        ksize = int(6*sigma + 1)
    
    kernel = cv2.getGaussianKernel(ksize,sigma)
    return convolution2D(img,kernel,kernel,border)
#function that subsamples an image
def subsample(img):
    return img[::2, ::2]

#function that applies the mask of dx and dy derivatives of an image
def maskDerivKernels(img,dx = 1,dy = 1,ksize = 3,border = cv2.BORDER_REPLICATE):
    """
    getDerivKernels returns coefficients to get images partial derivatives.
    dx and dy are respectively the partial orders for x and y.
    """
    dxdy = cv2.getDerivKernels(dx,dy,ksize,normalize = 0)
    return convolution2D(img,dxdy[0],dxdy[1],border)

#function that returns a gaussian pyramid from the original image
# iters are the number of levels, ksize is the kernel size and sigma is the gaussian blur sigma
def gaussianPyramid(orig,iters,ksize = 3,sigma1 = 1,border = cv2.BORDER_DEFAULT):
    pyramid = [orig]
    
    for i in range (0,iters+1):
        blurred = gaussian2D(pyramid[i],sigma1,ksize,border)
        subsampled = subsample(blurred)
        pyramid.append(subsampled)

    return pyramid
    

def supresionNoMax(im,winsize=3):
    
    #vim: conjunto de imágenes sobre la que se realiza la supresión
    filas=im.shape[0]
    columnas=im.shape[1]
    
    #realizamos una copia de la imagen
    supre=np.copy(im)
    #para cada pixel de la imagen, comprobamos si los pixeles adyacentes tienen un valor más alto, en cuyo caso ponemos a 0 el pixel en la copia
    for i in range(0,filas):
        for j in range(0,columnas):
            pixel=im[i][j]
            fil_inf=max(i-int((winsize-1)/2),0)
            fil_sup=min(i+int((winsize-1)/2)+1,filas)
            col_inf=max(j-int((winsize-1)/2),0)
            col_sup=min(j+int((winsize-1)/2)+1,columnas)
            for k in range(fil_inf,fil_sup):
                for l in range(col_inf,col_sup):
                    if im[k][l]>pixel:
                        supre[i][j]=0.0
    #devolvemos las imágenes suprimidas
    return supre


y1c = readIMG("datos-T2/yosemite/Yosemite1.jpg")
y1g = readIMG("datos-T2/yosemite/Yosemite1.jpg",0)
y2c = readIMG("datos-T2/yosemite/Yosemite2.jpg")
y2g = readIMG("datos-T2/yosemite/Yosemite2.jpg",0)

def corner_strength(l1,l2):
    if (l1+l2 == 0):
        return 0
    return (l1*l2)/(l1+l2)

def get_keypoints(img,block_size,level):
    dx = maskDerivKernels(img,1,0)
    dy = maskDerivKernels(img,0,1)

    dx = gaussian2D(dx,4.5)
    dy = gaussian2D(dy,4.5)
    keypoints = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i,j] > 0):
                ox = dx[i,j]
                oy = dy[i,j]
                cos,sen = ox/(math.sqrt(ox**2 + oy**2)),oy/(math.sqrt(ox**2 + oy**2))
                ori = math.atan2(sen,cos)*180/math.pi
                keypoints.append(cv2.KeyPoint(j*(2**level),
                                              i*(2**level),
                                              _size = block_size*(level+1),
                                              _angle = ori))
    return keypoints


def harris(src,level,block_size = 3,ksize = 3,threshold = 10):
    #Get lambda1,lambda2, eiv11,eiv12, eiv21,eiv22
    e_v = cv2.cornerEigenValsAndVecs(src,blockSize = block_size,ksize = ksize)
    #Corner strength matrix
    first_m = np.asarray([[ corner_strength(e_v[i,j,0],e_v[i,j,1]) 
                 for j in range(src.shape[1])] 
                 for i in range(src.shape[0])])
    # Get values that are > than threshold
    threshold_m = np.asarray([[ first_m[i,j] if first_m[i,j] > threshold else 0
                     for j in range(first_m.shape[1])]
                     for i in range(first_m.shape[0])])
    
    # Supress no max in winsize X winsize neighborhood
    sup_no_max_m  = supresionNoMax(threshold_m,5)
   
    # Return keypoints
    return get_keypoints(sup_no_max_m,block_size,level)

def ej1():
    # Get gaussian pyramid
    p1 = gaussianPyramid(y1c,iters = 3)

    all_keypoints = np.copy(y1c).astype(np.uint8)
    total_kp = 0

    for i in range(len(p1)):
        colorless = cv2.cvtColor(p1[i],cv2.COLOR_RGB2GRAY).astype(np.float32)
        kp = harris(colorless,i,block_size = 7,ksize = 3,threshold = 10)
        print("Found: " + str(len(kp)) + " keypoints at level " + str(i))
        total_kp += len(kp)
        # Draw circles
        copy = np.copy(y1c).astype(np.uint8)
        copy = cv2.drawKeypoints(copy,kp,np.array([]),flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS).astype(np.float64)
        all_keypoints = cv2.drawKeypoints(all_keypoints,kp,np.array([]),flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        pintaI(copy)
    
    pintaI(all_keypoints.astype(np.float64))
    print("Total keypoints: " + str(total_kp))
    


# In[52]:


def matchBruteForce(im1,im2):
    # Create Akaze descriptor
    akaze = cv2.AKAZE_create()
    # Get Akaze Descriptors and KP
    kp1,d1 = akaze.detectAndCompute(im1,None)
    kp2,d2 = akaze.detectAndCompute(im2,None)
    # Create BF MAtcher Object
    bfmatcher = cv2.BFMatcher.create(crossCheck = True)
    # Match
    matches = bfmatcher.match(d1,d2)
    
    return kp1,kp2,d1,d2,matches

def matchLoweAvg2NN(im1,im2,ratio = 0.7):
    # Create Akaze descriptor
    akaze = cv2.AKAZE_create()
    # Get Akaze Descriptors and KP
    kp1,d1 = akaze.detectAndCompute(im1,None)
    kp2,d2 = akaze.detectAndCompute(im2,None)
    # Create BF Matcher
    bfmatcher = cv2.BFMatcher.create()
    # Match using 2-nn
    matches = bfmatcher.knnMatch(d1,d2,k=2)
    # Get non ambiguous matches
    valid = []
    for m,n in matches:
        if m.distance < n.distance*ratio:
            valid.append([m])
    
    return kp1,kp2,d1,d2,valid
    


def ej2():
    # Vars
    n = 100
    im1 = y1c.astype(np.uint8)
    im2 = y2c.astype(np.uint8)
    # Brute Force Maches
    k1,k2,d1,d2,matches = matchBruteForce(im1,im2)
    # Get sample size = n
    sample = random.sample(matches,n)
    img = cv2.drawMatches(im1,k1,im2,k2,sample,None,flags = 2)
    pintaI(img.astype(np.float64))

    # LoweAvg2NN matches
    k1,k2,d1,d2,matches = matchLoweAvg2NN(im1,im2)
    # Get sample size = n
    sample = random.sample(matches,n)
    # Draw matches
    img = cv2.drawMatchesKnn(im1,k1,im2,k2,sample,None,flags = 2)
    pintaI(img.astype(np.float64))

#ej1()
#ej2()
# In[ ]:


def remove_extra(img):
    indexR =[]
    indexC = []
    for i in range(img.shape[0]):
        if np.count_nonzero(img[i]) == 0:
            indexR.append(i)
    for i in range(img.shape[1]):
        if np.count_nonzero(img[:,i]) == 0:
            indexC.append(i)
    img = np.delete(img,indexR,axis = 0)
    img = np.delete(img,indexC,axis = 1)
    return img

def getCanvas(imgs):
    return np.zeros((sum([img.shape[0] for img in imgs])*2,
                     sum([img.shape[1] for img in imgs])*2)).astype(np.uint8)

def identity_h(img,canvas):
    tx = canvas.shape[1]/2 - img.shape[1]/2
    ty = canvas.shape[0]/2 - img.shape[0]/2
    id = np.array([[1,0,tx],[0,1,ty],[0,0,1]],dtype = np.float32)
    return id

def homography(img1,img2):
    # Get matches
    k1,k2,d1,d2,matches = matchLoweAvg2NN(img1,img2)
    # Get and sort matching points
    orig = np.float32([k1[p[0].queryIdx].pt for p in matches]).reshape(-1,1,2)
    dest = np.float32([k2[p[0].trainIdx].pt for p in matches]).reshape(-1,1,2)
    # Get Homography. Using RANSAC
    h = cv2.findHomography(orig,dest,cv2.RANSAC,1)[0]

    return h

def two_mosaic(img1,img2):
    canvas = getCanvas([img1,img2])
    h = homography(img2,img1)
    id = identity_h(img1,canvas)
    # Introduce img1 in canvas
    canvas = cv2.warpPerspective(img1,id,(canvas.shape[1],canvas.shape[0]),dst = canvas,borderMode = cv2.BORDER_TRANSPARENT)
    comp = np.dot(id,h)
    canvas = cv2.warpPerspective(img2,comp,(canvas.shape[1],canvas.shape[0]),dst = canvas,borderMode = cv2.BORDER_TRANSPARENT)
    return canvas


def ej3():
    c = two_mosaic(y1c,y2c)
    c = remove_extra(c)

    pintaI(c)


def n_mosaic(imgs):
    # Get starting image and big canvas
    half = int(len(imgs)/2)
    canvas = getCanvas(imgs)
    # Get id homography for half image and warp it
    id = identity_h(imgs[half],canvas)
    canvas = cv2.warpPerspective(imgs[half],id,(canvas.shape[1],canvas.shape[0]),dst = canvas,borderMode = cv2.BORDER_TRANSPARENT)
    # Create vector for homographies and composition
    homs = [None]*len(imgs)
    homs[half] = id
    # Left Part
    for i in range(half)[::-1]:
        h_i = homography(imgs[i+1],imgs[i])
        #h_i = np.dot(homs[i+1],h_i)
        h_i = np.dot(h_i,homs[i+1])
        homs[i] = h_i
        canvas = cv2.warpPerspective(imgs[i],h_i,(canvas.shape[1],canvas.shape[0]),dst = canvas,borderMode = cv2.BORDER_TRANSPARENT)
   # Right Part
    for i in range(half,len(imgs)):
       h_i = homography(imgs[i],imgs[i-1])
       h_i = np.dot(h_i,homs[i-1])
       #h_i = np.dot(homs[i-1],h_i)
       homs[i] = h_i
       canvas = cv2.warpPerspective(imgs[i],h_i,(canvas.shape[1],canvas.shape[0]),dst = canvas,borderMode = cv2.BORDER_TRANSPARENT)

    return canvas

def ej4():
    y1 = readIMG("datos-T2/yosemite_full/yosemite1.jpg")
    y2 = readIMG("datos-T2/yosemite_full/yosemite2.jpg")
    y3 = readIMG("datos-T2/yosemite_full/yosemite3.jpg")
    y4 = readIMG("datos-T2/yosemite_full/yosemite4.jpg")
    y5 = readIMG("datos-T2/yosemite_full/yosemite5.jpg")
    y6 = readIMG("datos-T2/yosemite_full/yosemite6.jpg")
    y7 = readIMG("datos-T2/yosemite_full/yosemite7.jpg")

    e1 = readIMG("datos-T2/mosaico-1/mosaico002.jpg")
    e2 = readIMG("datos-T2/mosaico-1/mosaico003.jpg")
    e3 = readIMG("datos-T2/mosaico-1/mosaico004.jpg")
    e4 = readIMG("datos-T2/mosaico-1/mosaico005.jpg")
    e5 = readIMG("datos-T2/mosaico-1/mosaico006.jpg")
    e6 = readIMG("datos-T2/mosaico-1/mosaico007.jpg")
    e7 = readIMG("datos-T2/mosaico-1/mosaico008.jpg")
    e8 = readIMG("datos-T2/mosaico-1/mosaico009.jpg")
    e9 = readIMG("datos-T2/mosaico-1/mosaico010.jpg")
    e10 = readIMG("datos-T2/mosaico-1/mosaico011.jpg")
    c = n_mosaic([y1,y2,y3,y4,y5])
    c = remove_extra(c)
    pintaI(c)

    c2 = n_mosaic([e1,e2,e3,e4,e5,e6,e7,e8,e9,e10])
    c2 = remove_extra(c2)
    pintaI(c2)

#ej1()
#ej2()
#ej3()
ej4()
