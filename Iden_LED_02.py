#Importando as bibliotecas

import cv2 as cv
import numpy as np
import matplotlib.pyplot as mp

#Importando a imagem

img = cv.imread(r"/Users/PedroVitorPereira/PycharmProjects/CursodePython/Proj Identificacao LEDS IHM/Imagens IHM/IHM_00.jpeg")

##----------------------------------------------------------------------------------------------------------------------#

# Convertendo do espaço RGB para HSV

hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
# ----------------------------------------------------------------------------------------------------------------------#

##Mascara para as cores amarelas

amarelo_lower = np.array([10,100,120])
amarelo_upper = np.array([50,255,255])
mask_amarelo = cv.inRange(hsv, amarelo_lower,amarelo_upper)

#Detectanto e desenhando os Resultados amarelos
circles_amarelos = cv.HoughCircles(mask_amarelo,cv.HOUGH_GRADIENT,2, 20, param1=50, param2=30, minRadius=5,
                                     maxRadius=150)
circles_01 = np.uint16(np.around(circles_amarelos))

#Número de Resultados encontrados
a = np.shape(circles_01)

#Mostrando os resultados e salvando a imgem
cv.imshow('Circulos amarelos',mask_amarelo)
cv.imwrite("Resultados/Mascara Amarela.png", mask_amarelo)

cv.waitKey(0)
cv.destroyAllWindows()

cimg = img.copy()

for i in circles_01[0, :]:
    # draw the outer circle
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv.circle(cimg, (i[0], i[1]), 2, (255, 0, 0), 10)

cv.imshow('Leds Amarelo',cimg)
cv.imwrite("Resultados/Res_iden_amarelos.png", cimg)

cv.waitKey(0)
cv.destroyAllWindows()


# ----------------------------------------------------------------------------------------------------------------------#

# Mascara para as cores verdes

verde_lower = np.array([49, 50, 90])
verde_upper = np.array([80, 255, 255])
mask_verde = cv.inRange(hsv, verde_lower,verde_upper)

# Detectanto e desenhando os Resultados verdes
circles_verde = cv.HoughCircles(mask_verde,cv.HOUGH_GRADIENT,2, 20, param1=50, param2=30, minRadius=10,
                                     maxRadius=40)
circles_02 = np.uint16(np.around(circles_verde))

#print(circles_02)

# Número de Resultados encontrados
b = np.shape(circles_02)

#Verificando a dimensão dos raios

#for x in circles_02:
 #   print(x)

#Mostrando os resultados e salvando a imgem

cv.imshow('Circulos Verdes',mask_verde)
cv.imwrite("Resultados/Mascara Verdes.png", mask_verde)

cv.waitKey(0)
cv.destroyAllWindows()

cimg = img.copy()

for i in circles_02[0, :]:
    # draw the outer circle
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv.circle(cimg, (i[0], i[1]), 2, (255, 0, 0), 10)

cv.imshow('Leds Verdes',cimg)
cv.imwrite("Resultados/Res_iden_verdes.png", cimg)

cv.waitKey(0)
cv.destroyAllWindows()


##----------------------------------------------------------------------------------------------------------------------#

##Mascara para as cores vermelhas

vermelho_lower = np.array([150,100,120])
vermelho_upper = np.array([230,255,255])
mask_vermelho = cv.inRange(hsv, vermelho_lower,vermelho_upper)

#Detectanto e desenhando os Resultados vermelhos

circles_vermelho = cv.HoughCircles(mask_vermelho,cv.HOUGH_GRADIENT,2,15, param1=50, param2=30, minRadius=10,
                                     maxRadius=40)
circles_03 = np.uint16(np.around(circles_vermelho))

#Número de Resultados encontrados

c = np.shape(circles_03)

#Verificando a dimensão dos raios

#for x in circles_03:
 #   print(x)

#Mostrando os resultados e salvando a imgem

cv.imshow('Circulos Vermelhos',mask_vermelho)
cv.imwrite("Resultados/Mascara Vermelha.png", mask_vermelho)

cv.waitKey(0)
cv.destroyAllWindows()

cimg = img.copy()
for i in circles_03[0, :]:
    # draw the outer circle
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv.circle(cimg, (i[0], i[1]), 2, (255, 0, 0), 10)

cv.imshow('Leds Vermelhos',cimg)
cv.imwrite("Resultados/Res_iden_vermelho.png", cimg)

cv.waitKey(0)
cv.destroyAllWindows()


#----------------------------------------------------------------------------------------------------------------------#

##Mascara para as cores azuis

azul_lower = np.array([95,200,205])
azul_upper = np.array([145,255,255])
mask_azul = cv.inRange(hsv, azul_lower,azul_upper)

#Detectanto e desenhando os Resultados azuis
circles_azul = cv.HoughCircles(mask_azul,cv.HOUGH_GRADIENT,2, 20, param1=50, param2=30, minRadius=10,
                                     maxRadius=150)
circles_04 = np.uint16(np.around(circles_azul))

#Número de Resultados encontrados

d = np.shape(circles_04)

#Mostrando os resultados e salvando a imgem

cv.imshow('Circulos azuis',mask_azul)
cv.imwrite("Resultados/Mascara Azul.png", mask_azul)

cv.waitKey(0)
cv.destroyAllWindows()

cimg = img.copy()

for i in circles_04[0, :]:
    # draw the outer circle
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv.circle(cimg, (i[0], i[1]), 2, (255, 0, 0), 10)

cv.imshow('Leds Azuis',cimg)
cv.imwrite("Resultados/Res_iden_Azuis.png", cimg)

cv.waitKey(0)
cv.destroyAllWindows()

#----------------------------------------------------------------------------------------------------------------------#


# Resultados encontrados

relatorio = cv.putText(img,f"Foram identificados {a[1]} Resultados amarelos ",(10,30),cv.FONT_HERSHEY_TRIPLEX,1,200)
relatorio = cv.putText(img,f"Foram identificados {b[1]} Resultados verdes",(10,65),cv.FONT_HERSHEY_TRIPLEX,1,200)
relatorio = cv.putText(img,f"Foram identificados {c[1]} Resultados vermelhos ",(10,95),cv.FONT_HERSHEY_TRIPLEX,1,200)
relatorio = cv.putText(img,f"Foram identificados {d[1]} Resultados azuis",(10,125),cv.FONT_HERSHEY_TRIPLEX,1,200)

cv.imshow('Imagem Original',img)
cv.imwrite("Resultados/Resultado.png", img)
cv.waitKey(0)
cv.destroyAllWindows()
