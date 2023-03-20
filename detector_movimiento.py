import cv2
import numpy as np

cap = cv2.VideoCapture('C:/Users/Manue/Documents/pythonProject/tools/cuadro.mp4')
#cap = cv2.VideoCapture(0)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))


while True:

    ret, frame = cap.read()
    if ret == False: break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#escala de grises

    cv2.rectangle(frame, (0,0), (frame.shape[1],40),(0,0,0), -1) #dibujo del rectangulo
    color = (0, 255, 0) #color del rectangulo
    texto_estado = "M-bot: No hay alertas *.*"

    area_pts = np.array([[240,320],[480,320],[620,frame.shape[0]],[50,frame.shape[0]]])#especificamos los puntos extremos del area a analizar

    #actuador
    imAux = np.zeros(shape=(frame.shape[:2]),dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    imagen_area = cv2.bitwise_and(gray,gray,mask=imAux)

    #sustracción de fondo
    fgmask = fgbg.apply(imagen_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2) #cantidad de pixeles

    #cuando hay movimiento
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        if cv2.contourArea(cnt) > 500:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0),2)
            texto_estado = "M-Bot: Alerta de Movimiento -_-"
            color = (0,0,255) #Color rojo



    cv2.drawContours(frame, [area_pts], -1, color,2)#contornos del area que se va a nalizar y grosor de linea
    cv2.putText(frame, texto_estado,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)#texto y fuente del estado

    cv2.imshow('Py-Bot', frame)
    cv2.imshow('Py-Bot Binario', fgmask) #demostración de imagen binario 

    k = cv2.waitKey(80) & 0xFF
    if k==27: #scape
        break

cap.release()
cv2.destroyAllWindows()