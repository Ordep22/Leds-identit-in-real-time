import cv2 as cv
import numpy as np
import imutils


##Iniciando o captura de Imagens IHM pela WebCan

##O parâmetro que "0" pede que seja aberta a segunda WebCan (USB)

video_cap = cv.VideoCapture(0)

##Iniciando o laço de repetição para captura do frame
while True:

    ##Status retorno um valor booleano alto ou baixo se a captura foi iniciada ou não
    ##Já o parâmetro Frame é basicamente as imagem capturada pela câmera

    status, frame = video_cap.read()

    ##Transformando o espaço de cores de RGB para HSV

    ##A função cvtColor faz o processo de tranformação basicamente passando uma imagem e um método de conversão

    esp_hsv  = cv.cvtColor(frame, cv.COLOR_BGR2HSV)


    ##Iniciando o processo de criação de máscaras
##--------------------------------------------------------------------------------------------------------------------##

    ##Máscara para os LED's verdes

    verde_baixo = np.array([49, 50, 90])
    verde_alto = np.array([80, 255, 255])

    mascara_verde = cv.inRange(esp_hsv,verde_baixo,verde_alto)

    ##Identificando os contornos dos LEDS verdes

    contorno_verde = cv.findContours(mascara_verde,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    contorno_verde_01 = imutils.grab_contours(contorno_verde)

    for c in contorno_verde_01:
        area_01 = cv.contourArea(c)
        if area_01 > 250:
            cv.drawContours(frame,[c],-1,(30,255,255),3)
            M = cv.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv.putText(frame,"Led's verdes Identificados",(10,65),cv.FONT_HERSHEY_TRIPLEX, 1, 0)

##--------------------------------------------------------------------------------------------------------------------##

    ##Máscara para os LED's amrelo

    amarelo_baixo = np.array([10,100,120])
    amarelo_alto = np.array([50,255,255])

    mascara_amarelo = cv.inRange(esp_hsv, amarelo_baixo, amarelo_alto)


    ##Identificando os contornos dos LEDS amarelos

    contorno_amarelo = cv.findContours(mascara_amarelo, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contorno_amarelo_01 = imutils.grab_contours(contorno_amarelo)

    for d in contorno_amarelo_01:
        area_02 = cv.contourArea(d)
        if area_02 > 50:
            cv.drawContours(frame, [d], -1, (30, 255, 255), 3)
            M = cv.moments(d)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv.putText(frame, "Led's amrelos Identificados", (10, 95), cv.FONT_HERSHEY_TRIPLEX, 1, 0)

##--------------------------------------------------------------------------------------------------------------------##

        ##Máscara para os LED's vermelhos

        vermelho_baixo = np.array([150,50,120])
        vermelho_alto = np.array([230,255,255])

        mascara_vermelho = cv.inRange(esp_hsv, vermelho_baixo, vermelho_alto)

        ##Identificando os contornos dos LEDS vermelhos

        contorno_vermelho = cv.findContours(mascara_vermelho , cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contorno_vermelho_01 = imutils.grab_contours(contorno_vermelho)

        for e in contorno_vermelho_01:
            area_03 = cv.contourArea(e)
            if area_03 > 50:
                cv.drawContours(frame, [e], -1, (30, 255, 255), 3)
                M = cv.moments(e)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                cv.putText(frame, "Led's vermelhos Identificados", (10, 125), cv.FONT_HERSHEY_TRIPLEX, 1, 0)

##--------------------------------------------------------------------------------------------------------------------##
        ##Máscara para os LED's azuis

        azul_baixo = np.array([80,130,205])
        azul_alto = np.array([145,255,255])

        mascara_azul = cv.inRange(esp_hsv, azul_baixo, azul_alto)

        ##Identificando os contornos dos LEDS vermelhos

        contorno_azul = cv.findContours(mascara_azul, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contorno_azul_01 = imutils.grab_contours(contorno_azul)

        for e in contorno_azul_01:
            area_04 = cv.contourArea(e)
            if area_04 > 50:
                cv.drawContours(frame, [e], -1, (30, 255, 255), 3)
                M = cv.moments(e)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                cv.putText(frame, "Led's azuis Identificados", (10, 155), cv.FONT_HERSHEY_TRIPLEX, 1, 0)

##--------------------------------------------------------------------------------------------------------------------##



    ##Mostrando os resultado obtidos

    cv.imshow("Monitoramento em Tempo Real",frame)
    key = cv.waitKey(5)
    if key == 27:
        break


##Finalizando o processo de viasialição dos resultados

video_cap.release()
cv.destroyAllWindows()
