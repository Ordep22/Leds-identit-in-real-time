import cv2 as cv
import numpy as np
import imutils



vid_cap = cv.VideoCapture(0)

while True:
    valid, frame = vid_cap.read()


    #Transformando o espaço de cores
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    #Filtro para as cores verdes
    verde_lower = np.array([80,200,205])
    verde_upper = np.array([180,255,255])

    #Mascara para as cores verdes
    mask_verde = cv.inRange(hsv, verde_lower, verde_upper)

    #Encontrando os contornos das formas coloridas

    contorno_verde = cv.findContours(mask_verde, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contorno_verde_00 = imutils.grab_contours(contorno_verde)

    #print(contorno_verde)

    for c in contorno_verde_00:
        area = cv.contourArea(c)
        if area > 250:
            cv.drawContours(frame,[c],-1,(30,255,255),3)
            M = cv.moments(c)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv.circle(frame,(cx,cy),7,(255,255,255),-1)
            cv.putText(frame,"Led verde Identificado",(10,65),cv.FONT_HERSHEY_TRIPLEX,1,200)


    cv.imshow("Imagem WebCan", frame)
    key = cv.waitKey(5)
    if key  ==  27:
        break
#cv.imwrite("Frame.png",frame)

vid_cap.release()
cv.destroyAllWindows()



