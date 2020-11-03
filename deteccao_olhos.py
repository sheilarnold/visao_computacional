# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:09:59 2020

@author: SheilaCarolina
"""
import cv2

imagem = cv2.imread("C:\\Users\\SheilaCarolina\\Documents\\VisaoComputacional\\opencv\\pessoas\\beatles.jpg")
#imagem = cv2.imread("capitu.jpg")

img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
#
classificador_face = cv2.CascadeClassifier("C:\\Users\\SheilaCarolina\\Documents\\VisaoComputacional\\opencv\\cascades\\haarcascade_frontalface_default.xml")
classificador_olhos = cv2.CascadeClassifier("C:\\Users\\SheilaCarolina\\Documents\\VisaoComputacional\\opencv\\cascades\\haarcascade_eye.xml")

face_detectada = classificador_face.detectMultiScale(img_cinza)

for (x, y, l, a) in face_detectada:
    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
    regiao = imagem[y: y + a, x: x + l]
    regiao_cinza = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
    olhos_detectados = classificador_olhos.detectMultiScale(regiao_cinza, scaleFactor = 1.1, minNeighbors = 3)
    for (ox, oy, ol, oa) in olhos_detectados:
        cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (255, 0, 255), 2)

cv2.imshow("Face detectada", imagem)
cv2.waitKey()
