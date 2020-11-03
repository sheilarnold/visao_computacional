# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:08:25 2020

@author: SheilaCarolina
"""
import cv2

from datetime import datetime

video = cv2.VideoCapture(0)

classificador_face = cv2.CascadeClassifier("C:\\Users\\SheilaCarolina\\Documents\\VisaoComputacional\\opencv\\cascades\\haarcascade_frontalface_default.xml")
classificador_olhos = cv2.CascadeClassifier("C:\\Users\\SheilaCarolina\\Documents\\VisaoComputacional\\opencv\\cascades\\haarcascade_eye.xml")

dir_foto = "C:\\Users\\SheilaCarolina\\Documents\\VisaoComputacional\\opencv\\capturas"

while True:
    conectado, frame = video.read()
    
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detectada = classificador_face.detectMultiScale(frame_cinza, minSize = (150, 150))
    
    for(x, y, l, a) in face_detectada:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
    
    regiao = frame[y: y + a, x: x + l]
    regiao_cinza = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
    olhos_detectados = classificador_olhos.detectMultiScale(regiao_cinza, scaleFactor = 1.1, minNeighbors = 3)
    for (ox, oy, ol, oa) in olhos_detectados:
        cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (255, 0, 255), 2)
        #captura e armazenamento da imagem
        data = str(datetime.now())
        print(data)
        data = data.replace(" ", "_")
        print(data)
        new_dir_foto  = dir_foto + "\\captura{0}.jpg".format(data)
        print(new_dir_foto)
        print(cv2.imwrite("C:\\Users\\SheilaCarolina\\Documents\\VisaoComputacional\\opencv\\capturas\\imagems.jpg", frame))
        print(cv2.imwrite(new_dir_foto, frame))
    
    cv2.imshow("Video", frame)
    
    if(cv2.waitKey(1) == ord('q')):
        break

video.release()
cv2.destroyAllWindows()
