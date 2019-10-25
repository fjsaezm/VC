
---
title: "Cuestionario 1 - Visión por computador"
author: "Francisco Javier Sáez Maldonado"
header-includes:
  -  \usepackage[utf8]{inputenc}
  -  \usepackage[T1]{fontenc}
  -  \usepackage[sfdefault,scaled=.85, lining]{FiraSans}
  -  \usepackage{geometry}
  -  \geometry{left=3cm,right=3cm,top=3cm,bottom=3cm,headheight=1cm,headsep=0.5cm}

output:
    pdf_document
---

### Pregunta 1.- Diga en una sola frase cuál cree que es el objetivo principal de la Visión por Computador. Diga también cuál es la principal propiedad de las imágenes de cara a la creación algoritmos que la procesen.

El objetivo de la visión por computador es el uso de algoritmos que tratan imágenes y nos ayudan a obtener información sobre las mismas.

La propiedad principal que se se usa en los algoritmos es que dado un pixel, tiene un entorno de él en el que los valores de los píxeles vecinos son cercanos al del píxel en cuestión.
### Pregunta2.- Expresar las diferencias y semejanzas entre las operaciones de correlación y convolución. Dar una interpretación de cada una de ellas que en el contexto de uso en visión por computador.

Ambas operaciones utilizan una máscara $H$. Ambas son invariantes por transformaciones lineales. Sin embargo, la fórmula que nos da cada una es diferente, pues en la correlación viene dada por:
$$
G[i,j] = \sum_{u = -k}^k \sum_{v = -k}^k H[u,v] F[i+u,j+v]
$$
Mientras que en la convolución cambia ligeramente de la forma:
$$
G[i,j] = \sum_{u = -k}^k \sum_{v = -k}^k H[u,v] F[i-u,j-v]
$$
De hecho, cuando hacemos una convolución, lo que estamos haciendo es dar primero la vuelta por filas y por columnas a la máscara y luego aplicar una correlación.

En la visión por computador, la **correlación** se utiliza mas para encontrar patrones dentro de una imagen y la **convolución** se suele utilizar para aplicar filtros o suavizados a imágenes o para encontrar gradientes en la imagen
### Pregunta 3.- ¿Cuál es la diferencia “esencial” entre el filtro de convolución y el de mediana? Justificar la respuesta.
La diferencia esencial es que la convolución es lineal sobre las imágenes, pero el filtro de mediana no lo es, así que tendríamos que si $A,B$ son dos imágenes

### Pregunta 4.- Identifique el “mecanismo concreto” que usa un filtro de máscara para transformar una imagen.

El mecanismo que utiliza un filtro de máscara para transformar una imagen es el uso del vecindario (que puede ser de mayor o menor tamaño) de cada pixel para hacer la transformación de una imagen. Para ello, se toma como vecindario una submatriz de la original y se realizan operaciones con los vecinos del pixel en cuestión y con él mismo, según el filtro.


### Pregunta 5.- ¿De qué depende que una máscara de convolución pueda ser implementada por convoluciones 1D? Justificar la respuesta.
**Proposición.-** Sea $H$ la matriz de una convolución. Sea $H = \sum_i \sigma_i u_i v_i^T$ la descomposición SVD de la matriz $H$. Entonces, la convolución puede ser implementada por convolución 1D si $\sigma_0 \ne 0$


### Pregunta 6.- Identificar las diferencias y consecuencias desde el punto de vista teórico y de la implementación entre: a) Primero alisar la imagen y después calcular las derivadas sobre la imagen alisada b) Primero calcular las imágenes derivadas y después alisar dichas imágenes. Justificar los argumentos.


Teóricamente no hay ninguna diferencia entre estos dos procedimientos, pues la convolución tiene la propiedad conmutativa, así que si $F$ es la imagen, $M$ es la matriz del filtro de derivada respecto de alguno de los ejes, y $H$ es el filtro de alisamiento, se tiene que:
$$
M \star (H \star F) = H \star ( M \star F)
$$

En la implementación, si primero calculásemos las derivadas, luego tendríamos que alisar dos veces, una para cada uno de los ejes, así que es mejor alisar primero la imagen.

### Pregunta 7.- Identifique las funciones de las que podemos extraer pesos correctos para implementar de forma eficiente la primera derivada de una imagen. Suponer alisamiento Gaussiano.

### Pregunta 8.- Identifique las funciones de las que podemos extraer pesos correctos para implementar de forma eficiente la Laplaciana de una imagen. Suponer alisamiento Gaussiano.

### Pregunta 9.- Suponga que le piden implementar de forma eficiente un algoritmo para el cálculo de la derivada de primer orden sobre una imagen usando alisamiento Gaussiano. Enumere y explique los pasos necesarios para llevarlo a cabo.

### Pregunta 10.- Identifique semejanzas y diferencias entre la pirámide gaussiana y el espacio de escalas de una imagen, ¿cuándo usar una u otra? Justificar los argumentos.

### Pregunta 11.- ¿Bajo qué condiciones podemos garantizar una perfecta reconstrucción de una imagen a partir de su pirámide Laplaciana? Dar argumentos y discutir las opciones que considere necesario.

### Pregunta 12.-. ¿Cuáles son las contribuciones más relevantes del algoritmo de Canny al cálculo de los contornos sobre una imagen? ¿Existe alguna conexión entre las máscaras de Sobel y el algoritmo de Canny? Justificar la respuesta

### Pregunta 13.- Identificar pros y contras de k-medias como mecanismo para crear un vocabulario visual a partir del cual poder caracterizar patrones. ¿Qué ganamos y que perdemos? Justificar los argumentos

### Pregunta 14.- Identifique pros y contras del modelo de “Bolsa de Palabras” como mecanismo para caracterizar el contenido de una imagen. ¿Qué ganamos y que perdemos? Justificar los argumentos.


### Pregunta 15.- Suponga que dispone de un conjunto de imágenes de dos tipos de clases bien diferenciadas. Suponga que conoce como implementar de forma eficiente el cálculo de las derivadas hasta el orden N de la imagen. Describa como crear un algoritmo que permita diferenciar, con garantías, imágenes de ambas clases. Justificar cada uno de los pasos que proponga.
