import cv2
import numpy as np
from matplotlib import pyplot as plt


#General functions ----------------------------------------------

def BRG2RGB(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def onlyRead(filename,flagColor):
    return cv2.imread(filename,flagColor)

def changePixelValue(image,pixel,newValue):

    image[pixel[1]][pixel[0]] = newValue
    return image

#Found on: https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/. Concats readjusting size
def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

#Exercises ---------------------------------------------

def leeimagen(filename,flagColor):
    image = onlyRead(filename,flagColor)
    #image = BRG2RGB(image)
    cv2.imshow(filename, image)
    cv2.waitKey(0)

    return image

def pintaI(im,title = "img"):

    cv2.imshow(title,im)
    cv2.waitKey(0)

def pintaMI(vim,title = "imgs"):

    #Make all images BGR, so that they can all be displayed
    for i,im in enumerate(vim):
        if len(im.shape) == 2:
            vim[i] = cv2.cvtColor(vim[i],cv2.COLOR_GRAY2BGR)

    #Use found function to resize
    images = hconcat_resize_min(vim)
    pintaI(images,title)


#It wont change original picture
def modifyPixelsRandom(image,pixels):
    #Lets suppose the image is already read

    #wont change original picture
    new = np.copy(image)
    for pixel in pixels:
        #Random color for the pixel
        new = changePixelValue(new,pixel,np.random.random()*1000)

    return new


def pintaMITitles(vim,titles):
    title = " - ".join(titles)
    pintaMI(vim,title)


#Reading image flags can be used like:
# -  0 is for gray-scale (white and black)
# -  1 is for colours

#Ejercicio 1

img = leeimagen("images/orapple.jpg",0)
img2 = leeimagen("images/orapple.jpg",1)
cv2.destroyAllWindows()

#Ejercicio 2

matrix = np.random.random((100, 100))*956
pintaI(matrix)
#Normalize
matrix = cv2.normalize(matrix, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
pintaI(matrix)
cv2.destroyAllWindows()

#Ejercicio 3

"""Si no transformamos las imágenes, obtenemos un error en la función hconcat de esta llamada. El error es:
#src[i].dims <= 2 && src[i].rows == src[0].rows && src[i].type() == src[0].type() in function 'hconcat' """

pintaMI([img,img2])
cv2.destroyAllWindows()
img3 = onlyRead("images/logoOpenCV.jpg",0)
#img3 = BRG2RGB(img3)
pintaMI([img2,img3])
cv2.destroyAllWindows()

#Ejercicio 4

vector = []
for i in range(200):
    vector.append([i,i])

modify = modifyPixelsRandom(img, vector)
cv2.imshow('mod', modify)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Ejercicio 5

pintaMITitles([img,img2,img3,modify],["img1","img2","img3","modify"])
