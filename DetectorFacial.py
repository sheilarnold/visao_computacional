# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 19:41:39 2020

@author: SheilaCarolina
"""
import cv2

webcam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("C:\\Users\\SheilaCarolina\\Documents\\VisaoComputacional\\DetectorFacial\\opencv\\data\\haarcascades\\haarcascade_frontalface_default.xml")

while(True):
    operacao, video = webcam.read()
    video = cv2.flip(video, 180)
    faces = face_cascade.detectMultiScale(video, minNeighbors = 20, minSize = (30, 30), maxSize = (400, 400))
    
    for(x, y, w, h) in faces:
        cv2.rectangle(video, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.imshow("Face Detectada", video)
        if(cv2.waitKey(1) and 0xFF == ord('q')):#Se a techa 'q' for pressionada
            break#o sistema é parado

webcam.release()#Liberando a webcam
cv2.destroyAllWindows()#Fechando todas as janelas
