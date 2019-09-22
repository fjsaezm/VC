import cv2
from matplotlib import pyplot as plt
import numpy as np


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def leeimagen(filename,flagColor):
    image = cv2.imread(filename,flagColor)

    #Converts BRG to RGB, important
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    return image

def pintaI(im):

    plt.imshow(im)
    plt.show()
    plt.waitKey(0)

def pintaMI(vim):

    return 0



#Flags can be used like:
# -  0 is for gray-scale (white and black)
# -  1 is for colours

#Ejercicio 1
img = leeimagen("images/orapple.jpg",0)
img = leeimagen("images/orapple.jpg",1)


#Ejercicio 2
matrix = np.random.random((100, 100))*956
print(matrix)
matrix = matrix / np.linalg.norm(matrix)
print(matrix)
pintaI(matrix)
