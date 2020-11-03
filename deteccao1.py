# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:45:01 2020

@author: SheilaCarolina
"""
import cv2

imagem = cv2.imread("capitu.jpg")
img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", imagem)
cv2.imshow("Em escala de cinza", img_cinza)
cv2.waitKey();
