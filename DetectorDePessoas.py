# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 20:28:54 2020

@author: SheilaCarolina

OBS: É necessário utilizar o corpus INRIA person, o corpus atual tem +- 1000 imagens para treinamento e 1000 imagens para testes, de imagens com e sem pessoas.
    O ideal é que seja usado 100 imagens de cada tipo mas, se o computador não suportar tratar e carregar todas as imagens, 10 já são suficientes.
"""
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def get_descritores(img_caminho):
    largura = altura = 360
    img_teste = cv2.imread(img_caminho, 0)
    img_redimensionada = cv2.resize(img_teste,(largura, altura), interpolation=cv2.INTER_CUBIC)
    img_equalizada = cv2.equalizeHist(img_redimensionada)
    img_suavizada = cv2.GaussianBlur(img_equalizada, (9, 9), 1)
    orb = cv2.ORB_create(nfeatures = 512)
    pontos_chave = orb.detect(img_suavizada, None)
    pontos_chave, descritores = orb.compute(img_suavizada, pontos_chave)
    return descritores

class PacoteDePalavras:
    def gerar_dicionario(self, lista_descritores):
        print("Linha 1")
        kmeans = KMeans(n_clusters = 512, random_state = 0)
        print("Linha 2")
        print(kmeans)
        kmeans = kmeans.fit(lista_descritores)
        print("Linha 3")
        print(kmeans)
        self.dicionario = kmeans.cluster_centers_

    def histograma_de_frequencia(self, descritor):
        try:
            alg_knn = NearestNeighbors(n_neighbors = 1)
            alg_knn.fit(self.dicionario)
            proximos = alg_knn.kneighbors(descritor)
            hist_caracteristicas = np.histogram(proximos, bins=np.arange(self.dicionario.shape[0] + 1))[0]
            return proximos, hist_caracteristicas
        except AttributeError:
            print("Ocorreu um erro ao tratar o dicionário, tente novamente mais tarde.")

    def dict_save(self, caminho, dict_name):
        try:
            np.savetxt(os.path.join(caminho, dict_name), self.dicionario, delimiter = ',', fmt = '%f')
            print("Dicionário salvo com sucesso")
        except AttributeError:
            print("Erro ao salvar dicionário")

    def carregar_dicionario(self, caminho, dict_name):
        self.dicionario = np.loadtxt(os.path.join(caminho, dict_name), delimiter = ',')

dados_treinamento = ["C:\\Users\\SheilaCarolina\\Documents\\VisaoComputacional\\dadosImagem\\Treinamento\\pos\\", "C:\\Users\\SheilaCarolina\\Documents\\VisaoComputacional\\dadosImagem\\Treinamento\\neg\\"]
list_descritores = np.empty((0, 32), dtype = np.uint8)

for caminho in dados_treinamento:
    for dir_raiz, diretorios, arquivos in os.walk(caminho):
        for arquivo in arquivos:
            if arquivo.endswith('.png'):
                orb_descritor = get_descritores(os.path.join(caminho, arquivo))
                list_descritores = np.append(list_descritores, orb_descritor, axis = 0)
                print("Descritor armazenado.")
            print(arquivo + " lido.")
    print("***************" + caminho + "carregado.***************")

#img_caminho = "C:\\Users\\SheilaCarolina\\Documents\\VisaoComputacional\\dadosImagem\\Treinamento\\pos\\crop001011.png"
#descritor = get_descritores(img_caminho)
#teste_palavras_virtuais = PacoteDePalavras()
#teste_palavras_virtuais.gerar_dicionario(descritor)


img_representacao = PacoteDePalavras()
img_representacao.gerar_dicionario(list_descritores)
img_representacao.dict_save("C:\\Users\\SheilaCarolina\\Documents\\VisaoComputacional\\dicionarios\\", 'dicionario.csv')

def salvar_descritor(descritor, caminho, nome_arquivo):
#    print(descritor)
#    descritor1 = descritor.reshape((1, descritor.size))
#    descritor1 = np.array(descritor)
#    descritor1 = np.asarray((1, descritor))
    descritor1 = np.shape((1, np.array(descritor)))
    arquivo = open(os.path.join(caminho, nome_arquivo), 'a')
    np.savetxt(arquivo, descritor1, delimiter = ',', fmt = '%s')
    arquivo.close()

#Computar descritores gerando histograma de cada imagem separada
for caminho in dados_treinamento:
    print('1')
    for raiz, diretórios, arquivos in os.walk(caminho):
        print('2')
        for arquivo in arquivos:
            print('3')
            if arquivo.endswith('.png'):
                print('4')
                descritor = get_descritores(os.path.join(caminho, arquivo))
                print('5')
                histograma_descritor = img_representacao.histograma_de_frequencia(descritor)
                print('6')
                print(caminho)
                salvar_descritor(histograma_descritor, caminho, 'descritor.csv')
print("Extração de caracteristicas realizada e descritores salvos")
