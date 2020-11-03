# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 18:09:18 2020

@author: SheilaCarolina
"""
import cv2

imagem = cv2.imread("C:\\Users\\SheilaCarolina\\Documents\\VisaoComputacional\\opencv\\pessoas\\pessoas3.jpg")
#imagem = cv2.imread("capitu.jpg")
#cv2.imshow("Original", imagem)

img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Em escala de cinza", img_cinza)

classificador = cv2.CascadeClassifier("C:\\Users\\SheilaCarolina\\Documents\\VisaoComputacional\\opencv\\cascades\\haarcascade_frontalface_default.xml")
faces_detectadas = classificador.detectMultiScale(img_cinza, scaleFactor = 1.1, minNeighbors=9, minSize=(50, 50))

for (x, y, l, a) in faces_detectadas:
    cv2.rectangle(imagem, (x, y), (x + l, y + a),(0, 0, 255), 2 )

cv2.imshow("Faces detectadas", imagem)
cv2.waitKey()

