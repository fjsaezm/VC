#!/usr/bin/env python
# coding: utf-8
# Author: Francisco Javier Sáez Maldonado


import cv2
import numpy as np
from matplotlib import pyplot as plt

#function that turns BRG image to RGB
def BRG2RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#function that reads an img
def readIMG(filename,flagColor = 1):
        if flagColor == 0:
            return cv2.imread(filename,flagColor).astype(np.float64)
        else:
            return BRG2RGB(cv2.imread(filename,flagColor)).astype(np.float64)
    
#Found on: https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/. Concats readjusting size
#function that concats images with different sizes
def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

#function that normalizes a matrix
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

#function that draws an img on screen
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

#function that prints multiple images with same size
def pintaMI(vim,title = "imgs"):

    vim = [normalizeM(i) for i in vim]
    #Pasa a "formato color" las imágenes que estén en blanco y negro
    for i,im in enumerate(vim):
        if len(im.shape) == 2:
            vim[i] = cv2.cvtColor(vim[i].astype(np.uint8),cv2.COLOR_GRAY2BGR).astype(np.float64)
    
    #Use found function to resize
    images = hconcat_resize_min(vim)
    pintaI(images,title)

#function that prints a vector of images with different size keeping the different size
def pintaMIReduced(vim,title = "imgs"):
    
    copy = np.copy(vim)
    copy = [normalizeM(i) for i in copy]
    wmax = max(im.shape[1] for im in copy[1:len(copy)-1])
    for i,im in enumerate(copy):
        if len(im.shape) == 2:
            copy[i] = cv2.cvtColor(copy[i].astype(np.uint8),cv2.COLOR_GRAY2BGR).astype(np.float64)
        if im.shape[1] < wmax:
            copy[i] = cv2.copyMakeBorder(copy[i], 0,0,0,wmax - im.shape[1],borderType= cv2.BORDER_CONSTANT)
    
    im = cv2.vconcat(copy[1:len(copy)-1])
    if im.shape[0] > copy[0].shape[0]:
        copy[0] = cv2.copyMakeBorder(copy[0],0,im.shape[0]- copy[0].shape[0],0,0,borderType = cv2.BORDER_CONSTANT)
    else:
        im      = cv2.copyMakeBorder(im,0,copy[0].shape[0]-im.shape[0],0,0,borderType = cv2.BORDER_CONSTANT)
    images = cv2.hconcat([copy[0],im])
    pintaI(images,title)

#function that prints a couple of images and joins a vector of titles to print it
def pintaMITitles(vim,titles):
    title = " - ".join(titles)
    pintaMI(vim,title)
    

#read imgs
bird = normalizeM(readIMG('imagenes/bird.bmp'))
cat  = normalizeM(readIMG('imagenes/cat.bmp'))
mari = normalizeM(readIMG('imagenes/marilyn.bmp',0))
dog = normalizeM(readIMG('imagenes/dog.bmp',1))
einst = normalizeM(readIMG('imagenes/einstein.bmp',0))
plane = normalizeM(readIMG('imagenes/plane.bmp',1))

bwdog = readIMG('imagenes/dog.bmp',0)
bwcat = readIMG('imagenes/cat.bmp',0)
bwplane = readIMG('imagenes/plane.bmp',0)
bwbird  = readIMG('imagenes/bird.bmp',0)




#pintaMITitles([bird,cat,mari,dog,einst,plane],["bird","cat","marilyn","dog","einstein","plane"])

#function that makes a convolution given 2 kernels and bordertype to an img
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

#function that applies a gaussian2D filter to an img, given a sigma and a kernel size
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

#function that prints on screen the first part of exercise 1a
def ejercicio1a():
    a = gaussian2D(bird,5,border = cv2.BORDER_CONSTANT)
    b = gaussian2D(bird,22,7,border = cv2.BORDER_ISOLATED)
    c = gaussian2D(bird,22,31,border = cv2.BORDER_ISOLATED)
    d = gaussian2D(bird,22,border = cv2.BORDER_ISOLATED)
    pintaI(a,"Gaussian with s = 5")
    pintaMITitles([b,c,d], ["Gaussian with s = 22, ksize = 7", "Gaussian with s = 22, ksize = 31","Gaussian with s=22, kernel = auto"])


#function that applies the mask of dx and dy derivatives of an image
def maskDerivKernels(img,dx = 1,dy = 1,ksize = 3,border = cv2.BORDER_REPLICATE):
    """
    getDerivKernels returns coefficients to get images partial derivatives.
    dx and dy are respectively the partial orders for x and y.
    """
    dxdy = cv2.getDerivKernels(dx,dy,ksize,normalize = 0)
    return convolution2D(img,dxdy[0],dxdy[1],border)

#function that prints on screen the exercise 1a the second part
def ejercicio1a2():
    imgDeriv1 = maskDerivKernels(bird)
    imgDeriv2 = maskDerivKernels(bird,3,3,9)
    imgDeriv3 = maskDerivKernels(bird,6,2,9)
    imgDeriv4 = maskDerivKernels(bird,2,6,9)

    pintaMITitles([imgDeriv1,imgDeriv2],["First derivative each axis","Forth derivative each axis"])
    pintaMITitles([imgDeriv3,imgDeriv4],["Dx = 6,Dy = 2", "Dx = 2, Dy = 6"])


#function that returns the laplacian of an image
def laplacian(img,ksize):
    d2x   = maskDerivKernels(img,2,0,ksize)
    d2y   = maskDerivKernels(img,0,2,ksize)
    return d2x + d2y

#function that returns the laplacian of a gaussian from a image
def laplacian_gaussian(img,sigma1,kGaussian,kLaplacian,border = cv2.BORDER_DEFAULT):
    gauss = gaussian2D(img,sigma1,kGaussian,border)
    lap = laplacian(gauss,kLaplacian)
    return lap

#function that gives the result for exercise 1b
def ejercicio1b():
    lapgau1 = laplacian_gaussian(cat,1,33,5)
    lapgau2 = laplacian_gaussian(cat,3,33,5)

    lapgau3 = laplacian_gaussian(bird,1,99,3,cv2.BORDER_ISOLATED)
    lapgau4 = laplacian_gaussian(bird,1,99,3,cv2.BORDER_REPLICATE)


    pintaMI([lapgau1,lapgau2], "s = 1, ksizeGaussian = 33  -  s = 3, ksizeGaussian = 33")
    pintaMI([lapgau3,lapgau4], "BORDER_ISOLATED - BORDER_REPLICATE")


#function that subsamples an image
def subsample(img):
    return img[::2, ::2]

#function that upsamples an image
def upsample(img):
    i = [2*a for a in range(0,img.shape[0])]
    j = [2*a for a in range(0,img.shape[1])]
    for a in i:
        img = np.insert(img,a+1,img[a,:],axis = 0)
        
    for a in j:
        img = np.insert(img,a+1,img[:,a],axis = 1)
    
    img = gaussian2D(img,1,3,cv2.BORDER_REFLECT)
        
    return img


#function that returns a gaussian pyramid from the original image
# iters are the number of levels, ksize is the kernel size and sigma is the gaussian blur sigma
def gaussianPyramid(orig,iters,ksize,sigma1,border = cv2.BORDER_DEFAULT):
    pyramid = [orig]
    
    for i in range (0,iters+1):
        blurred = gaussian2D(pyramid[i],sigma1,ksize,border)
        subsampled = subsample(blurred)
        pyramid.append(subsampled)

    return pyramid


#declare pyramids outside function because they're reused
pyramid = gaussianPyramid(bird,4,0,2)
pyramid2 = gaussianPyramid(bird,4,0,3,cv2.BORDER_CONSTANT)
pyramid3 = gaussianPyramid(cat,4,0,9,cv2.BORDER_REPLICATE)

#function that prints the exercise 2a
def ejercicio2a():

    pintaMIReduced(pyramid,"auto kernel size")
    pintaMIReduced(pyramid2, "border constant")
    pintaMIReduced(pyramid3, "border replicate,  auto kernel size, sigma = 9")

#function that removes a row or col from the upsampled image 
def rmRowCol(up, img):
    # Check rows
    if up.shape[0] > img.shape[0]:
        up = up[:up.shape[0]-1,::]
    #Check cols
    if up.shape[1] > img.shape[1]:
        up = up[::,:up.shape[1]-1]
    return up

#Function that returns a laplacian pyramid
def laplacianPyramid(gaussPyramid):
    gaussPyramid = gaussPyramid[::-1]
    pyramid = [gaussPyramid[0]]
    levels = len(gaussPyramid)-1
    for i in range(0,levels):
        up = upsample(gaussPyramid[i])
        #Check if they have the same dimensions
        if up.shape != gaussPyramid[i+1].shape:
            up = rmRowCol(up,gaussPyramid[i+1])
        #Create level of Laplacian Pyramid
        level = np.subtract(up,gaussPyramid[i+1])
        pyramid.append(level)
    
    return pyramid[::-1]

#Function that makes exercise 2b
def ejercicio2b():
    lap_pyramid = laplacianPyramid(pyramid)
    lap_pyramid_borderC = laplacianPyramid(pyramid2)
    pintaMIReduced(lap_pyramid_borderC, "Laplacian pyramid using border constant gaussian Pyramid")


#cv.Circle(img, center, radius, color, thickness=1, lineType=8, shift=0) → None
#Function that draws circles in pixels that have bigger value than scale
def drawCircles(image,sigma,k,scale):
    index = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > scale:
                index.append((j,i))

    for i in index:
        image = np.copy(cv2.circle(image,i,int(k*sigma),255,thickness = 1))
        
    return image

#Function that supresses the no maximum items and returns a copy with them supressed
def supressNoMax(img):
    #Copy the image, and add 2 new rows and 2 new rows in both sides, all zeros
    ret = np.copy(img)
    m = np.copy(img)
    m = np.insert(m,0,0,axis = 0)
    m = np.insert(m,m.shape[0],0,axis = 0)
    m = np.insert(m,0,0,axis = 1)
    m = np.insert(m,m.shape[1],0,axis = 1)

    #Supress no max
    for x in range(1,m.shape[0]-1):
        for y in range(1,m.shape[1]-1):
            n = np.copy(m[x-1:x+2,y-1:y+2])
            n[1][1] = 0
            vmax = np.max(n)
            if vmax >= m[x][y]:
                ret[x-1][y-1] = 0
            else:
                ret[x-1][y-1] = m[x][y]  
    return ret


# Function that returns the laplacian scales space and the images with the no max supression
def laplacianScaleSpace(img,n,k):
    sigma = 1
    scales = []
    nomax  = []
    
    for i in range(0,n):
        #Calculate Laplacian of gaussian and normalize in scale
        lgn = laplacian_gaussian(img,sigma,9,9)*sigma*sigma
        #Save the m^2
        squared = np.power(lgn,2)
        scales.append(squared)
        #update sigma
        sigma = sigma*k
        #first supressNoMax, then normalize
        supressed = normalizeM(supressNoMax(squared))
        nomax.append(supressed)
    
    return scales,nomax

#Function that makes exercise 2c
def ejercicio2c():
    s,n   = laplacianScaleSpace(mari,4,1.2)
    for i,im in enumerate(n):
        n[i] = drawCircles(im,np.power(1.3,i),3,100)

    bwcat = readIMG('imagenes/cat.bmp',0)
    s1,n1 = laplacianScaleSpace(bwcat,4,1.3)
    for i,im in enumerate(n1):
         n1[i] = drawCircles(im,np.power(1.3,i),20,100)

    pintaMI(s1, "Cat Laplacian Scale Space")
    pintaMI(n1, "Regions detected in cat LSS")
    pintaMI(s, "Marilyn Laplacian Scale Space")
    pintaMI(n, "Regions detected in Marilyn LSS")


#Function that returns hybrid image with img1 and img2
def hybridImage(img1,img2,s1,s2):
    high = normalizeM(gaussian2D(img1,s1))
    low  = img2 - normalizeM(gaussian2D(img2,s2))
    return high + low


# Function that makes the process of exercise 3
def ejercicio3():

    hybrid1 = hybridImage(bwdog,bwcat,8,8)
    scaleH1 = gaussianPyramid(hybrid1,4,9,1)
    
    hybrid2 = hybridImage(mari,einst,5,10)
    scaleH2 = gaussianPyramid(hybrid2,4,9,1)

    hybrid3 = hybridImage(bwplane,bwbird,10,24)
    scaleH3 = gaussianPyramid(hybrid3,4,9,1)

    pintaMI([bwcat,bwdog,hybrid1], "Cat -  Dog - Hybrid")
    pintaMI([mari,einst,hybrid2], "Marylin - Einstein - Hybrid")
    pintaMI([bwplane,bwbird,hybrid3], "Plane - Bird - Hybrid")

    # Pyramids
    pintaMIReduced(scaleH1,"Cat => Dog")
    pintaMIReduced(scaleH2,"Einstein => Marilyn")
    pintaMIReduced(scaleH3,"Seagull => Plane")

#Variables and prints in the bonus 2 exercise
def bonus2():

    moto = readIMG('imagenes/motorcycle.bmp')
    bici = readIMG('imagenes/bicycle.bmp')

    hybrid1 = hybridImage(moto,bici,8,8)
    scaleH1 = gaussianPyramid(hybrid1,4,9,1)

    hybrid2 = hybridImage(dog,cat,5,10)
    scaleH2 = gaussianPyramid(hybrid2,4,9,1)

    hybrid3 = hybridImage(plane,bird,10,24)
    scaleH3 = gaussianPyramid(hybrid3,4,9,1)

    pintaMI([moto,bici,hybrid1], "Moto -  Bike - Hybrid")
    pintaMI([dog,cat,hybrid2], "Dog - Cat - Hybrid")
    pintaMI([bird,plane,hybrid3], "Bird- Plane - Hybrid")


    # Pyramids
    pintaMIReduced(scaleH1,"Bike => Moto")
    pintaMIReduced(scaleH2,"Cat => Dog")
    pintaMIReduced(scaleH3,"Seagull => Plane")



#Defines variables and prints the bonus 3
def bonus3():
    caja = readIMG('imagenes/caja.jpg', 1)
    cubo = readIMG('imagenes/rubik.jpg',1)
    pintaMI([caja,cubo],"Caja - Cubo")

    h = hybridImage(caja,cubo,5,5)
    pintaMI([caja,cubo,h],"Box - Rubik - Hybrid")
    scaleBonus = gaussianPyramid(h,4,9,1)
    pintaMIReduced(scaleBonus, "Gaussian Pyramid of the hybrid image")

# Wraps the 2 parts of exercise 1
def ej1a():
    print("Ejercicio1-a")
    ejercicio1a()
    print("Ejercicio1-a, segunda parte")
    ejercicio1a2()


def main():
    ej1a()
    print("Ejercicio 1-b")
    ejercicio1b()
    print("Ejercicio 2-a")
    ejercicio2a()
    print("Ejercicio 2-b")
    ejercicio2b()
    print("Ejercicio 2-c. (Quizá tarde en ejecutar)")
    ejercicio2c()
    print("Ejercicio 3")
    ejercicio3()

    print("Bonus 2")
    bonus2()

    print("Incluya las imágenes adjuntas en su carpeta imagenes o cambie la ruta en la funcion bonus3() ")
    print("Bonus 3")
    bonus3()

main()
