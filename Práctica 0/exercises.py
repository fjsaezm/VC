import cv2
from matplotlib import pyplot as plt
import numpy as np



def leeimagen(filename,flagColor):
    image = cv2.imread(filename,flagColor)

    #Converts BRG to RGB, important
    cv2.imshow('image', image)
    cv2.waitKey(0)

    return image

def pintaI(im):

    cv2.imshow('matrix',im)
    cv2.waitKey(0)

def pintaMI(vim):

    #Added because if it 
    #h = max(image.shape[0] for image in vim)

    for i,im in enumerate(vim):
        #im.shape contains rows, columns and channel (if image is color). if len = 2, it means there is no color.
        if len(im.shape)== 2:
            vim[i] = cv2.cvtColor(vim[i],cv2.COLOR_GRAY2BGR)

    #hconcat creates a new imagen concatenating a list of images
    images = cv2.hconcat(vim)
    pintaI(images)


#Flags can be used like:
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
matrix = matrix / np.linalg.norm(matrix)
pintaI(matrix)

cv2.destroyAllWindows()

#Ejercicio 3

#Si no transformamos las imágenes, obtenemos un error en la función hconcat de esta llamada. El error es:
#src[i].dims <= 2 && src[i].rows == src[0].rows && src[i].type() == src[0].type() in function 'hconcat'

pintaMI([img,img2])

cv2.destroyAllWindows()

img3 = leeimagen("images/logoOpenCV.jpg",0)

pintaMI([img,img3])
