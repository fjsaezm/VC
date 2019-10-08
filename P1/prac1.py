import cv2
import numpy as np
from matplotlib import pyplot as plt


def BRG2RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def readIMG(filename,flagColor = 1):
    return BRG2RGB(cv2.imread(filename,flagColor))

#Found on: https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/. Concats readjusting size
def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def pintaI(im,title = "img"):
    plt.title(title)
    plt.imshow(im)
    plt.axis('off')
    plt.show()

def pintaMI(vim,title = "imgs"):

    #Pasa a "formato color" las imágenes que estén en blanco y negro
    for i,im in enumerate(vim):
        if len(im.shape) == 2:
            vim[i] = cv2.cvtColor(vim[i],cv2.COLOR_GRAY2BGR)

    #Use found function to resize
    images = hconcat_resize_min(vim)
    pintaI(images,title)

img = readIMG('imagenes/bird.bmp')
blurred = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)

pintaMI([img,blurred])
 
