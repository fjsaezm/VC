#Visión por Computador
#Francisco Javier Sáez Maldonado


import cv2
import numpy as np
from matplotlib import pyplot as plt


#General functions ----------------------------------------------

def BRG2RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

def normalizeMatrix(m):

    #Floats
    m = m.astype(np.float64)

    #If its black and white
    if (len(m.shape)) == 2:
        max = np.max(m)
        min = np.min(m)
        if (max != min):
            m = (m - min)/(max - min)
        else :
            m = 0

    #If its colored
    else:
        maxs = np.max(m,(0,1))
        mins = np.min(m,(0,1))
        if maxs[0] != mins[0] and maxs[1] != mins[1]:
            m = (m - mins)/(maxs-mins)
        else:
            m = 0

    m = m*255
    #Back to uint
    m = m.astype(np.uint8)
    return m


#Exercises ------------------------------------------------------------

# Lee una imagen de un fichero en color o blanco y negro según el flag deseado
def leeimagen(filename,flagColor):
    #Leemos la imagen
    image = onlyRead(filename,flagColor)
    #Transformamos los colores de BRG a RGB para poder pintar las imágenes correctamente
    image = BRG2RGB(image)
    #Añadimos el título a la ventana
    plt.title(filename)
    #Dibujamos
    plt.imshow(image)
    plt.show()
    cv2.waitKey(0)

    return image


# Dibuja una matriz de colores con un título
def pintaI(im,title = "img"):
    plt.title(title)
    plt.imshow(im)
    plt.show()
    cv2.waitKey(0)


# Dibuja una lista de imágenes, todas en la misma ventana y con un título predeterminado si no se le pasa ningún argumento
def pintaMI(vim,title = "imgs"):

    #Pasa a "formato color" las imágenes que estén en blanco y negro
    for i,im in enumerate(vim):
        if len(im.shape) == 2:
            vim[i] = cv2.cvtColor(vim[i],cv2.COLOR_GRAY2BGR)

    #Use found function to resize
    images = hconcat_resize_min(vim)
    pintaI(images,title)


#Modifica de forma aleatoria una lista de píxeles que se le pasan por parámetro
#It wont change original picture
def modifyPixels(image,pixels,new_values = []):


    # Copiamos para no modificar la imagen original, devolvemos una nueva
    new = np.copy(image)

    if len(pixels) != len(new_values):
        for pixel in pixels:
            #Random color for the pixel
            new = changePixelValue(new,pixel,np.random.random()*1000)

    else:
        for i,pixel in enumerate(pixels):
            new = changePixelValue(new,pixel,new_values[i])

    return new



# Dibuja una lista de imágenes poniendo sus títulos separados por un guión
def pintaMITitles(vim,titles):
    title = " - ".join(titles)
    pintaMI(vim,title)


#Reading image flags can be used like:
# -  0 is for gray-scale (white and black)
# -  1 is for colours

#Ejercicio 1

img1 = leeimagen("images/orapple.jpg",0)
img2 = leeimagen("images/orapple.jpg",1)
img3 = onlyRead("images/logoOpenCV.jpg",0)

cv2.destroyAllWindows()

#Ejercicio 2

matrix = normalizeMatrix(np.random.rand(100,100))
matrix = BRG2RGB(matrix)
matrix_color = normalizeMatrix(np.random.rand(100,100,3))
matrix_color = BRG2RGB(matrix_color)
pintaI(matrix)
pintaI(matrix_color)

cv2.destroyAllWindows()

#Ejercicio 3

"""Si no transformamos las imágenes, obtenemos un error en la función hconcat de esta llamada. El error es:
#src[i].dims <= 2 && src[i].rows == src[0].rows && src[i].type() == src[0].type() in function 'hconcat' """

img2 = BRG2RGB(img2)
img2 = BRG2RGB(img2)
pintaMI([img1,img2])
cv2.destroyAllWindows()
pintaMI([img2,img3])
cv2.destroyAllWindows()

#Ejercicio 4

vector = [ [i,i] for i in range(200) ]
modify = modifyPixels(img1,vector)
modify = normalizeMatrix(modify)
pintaI(modify,'mod')
cv2.destroyAllWindows()


#Ejercicio 5

pintaMITitles([img1,img2,img3,modify],["img1","img2","img3","modify"])
