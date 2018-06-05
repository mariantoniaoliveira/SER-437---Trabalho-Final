#################### IMPORTANDO AS BIBLIOTECAS NECESSÁRIAS PARA O FUNCIONAMENTO DO CÓDIGO #######################

from osgeo import gdal, ogr

import geopandas as gpd

import numpy as np

# importar constantes
from gdalconst import *

# informar o uso de exceções
gdal.UseExceptions()

# biblioteca de funções relacionadas ao sistema
# sys: System-specific parameters and functions
import sys

import glob

from numpy import *
from numpy.linalg import inv

import os

import math

###################################### ACESSANDO OS DADOS EM FORMATO SHP ######################################

# informando diretório

dir_shp_tr = 'C:/Users/Maria/_trabalho_final_intro/amostras_shp/samples_tr.shp' # amostras de treinamento
dir_shp_tst = 'C:/Users/Maria/_trabalho_final_intro/amostras_shp/samples_tst.shp' # amostras de teste

#acessando as amostras treinamento e teste (apenas geometria do tipo Ponto)
shp_tr = ogr.Open(dir_shp_tr)
shp_tst = ogr.Open(dir_shp_tst)

layer_tr = shp_tr.GetLayer()
layer_tst = shp_tst.GetLayer()

# identificando o número total de classes 
tam_tr = layer_tr.GetFeatureCount()
tam_tst = layer_tst.GetFeatureCount()

# checando uma determinada posição de um objeto
atr_tr = layer_tr.GetFeature(5)
atr_tst = layer_tst.GetFeature(5)

# verificando as cooordenadas do objeto selecionado
nx = atr_tr.GetField("X")
ny = atr_tr.GetField("Y")

# identificando o ID
fid = atr_tr.GetFID()

###################################### TRATANDO OS DADOS EM FORMATO SHP ######################################
# checando o número de classes 
nclass = 1
for i in range(tam_tr):
    aux = layer_tr.GetFeature(i)
    auxclass = aux.GetFieldAsInteger64("num_classe")
    if auxclass > nclass:
        nclass = auxclass

# transformando shape em uma tabela
nlin_tr = layer_tr.GetFeatureCount()
nlin_tst = layer_tst.GetFeatureCount()

# construindo a matriz a partir do numpy para amostras de treinamento
matriz_shape_tr = np.zeros((nlin_tr, 4))

for i in range(nlin_tr):
    aux = layer_tr.GetFeature(i)
    matriz_shape_tr[i,0]= aux.GetFieldAsInteger("FID")
    matriz_shape_tr[i,1]= aux.GetFieldAsInteger64("num_classe")
    matriz_shape_tr[i,2]= aux.GetField("X")
    matriz_shape_tr[i,3]= aux.GetField("Y")

# construindo a matriz a partir do numpy para amostras de teste
matriz_shape_tst = np.zeros((nlin_tst, 4))

for i in range(nlin_tst):
    aux = layer_tst.GetFeature(i)  
    matriz_shape_tst[i,0]= aux.GetFieldAsInteger("FID")
    matriz_shape_tst[i,1]= aux.GetFieldAsInteger64("num_classe")
    matriz_shape_tst[i,2]= aux.GetField("X")
    matriz_shape_tst[i,3]= aux.GetField("Y")

###################################### SALVAR O RESULTADO EM TXT ######################################

num_arq = 0 # variável que modifica o nome do arquivo de saída.

# Acessando o diretório onde estão salvas as imagens
os.chdir(os.path.join(os.getcwd(), 'C:/Users/Maria/_trabalho_final_intro/imagens'))
os.getcwd()

# Criando lista com os nomes das imagens acessadas no diretório
lista_nomes = os.listdir(os.getcwd())
nome_arq = []

for dados in lista_nomes:
    if dados[-3:] == 'tif':
        nome_arq2 = nome_arq.append(dados[:-4])
        
################################## ACESSANDO AS IMAGENS NO DIRETÓRIO ##################################   

for img_file in glob.glob('C:/Users/Maria/_trabalho_final_intro/imagens/*.tif'):
    
    # lendo a imagem
    img = gdal.Open(img_file, GA_ReadOnly)

    # número de linhas e colunas
    linhas = img.RasterYSize
    colunas = img.RasterXSize

    # quantidade de bandas
    nband = img.RasterCount

    # acessando as bandas
    band = img.GetRasterBand(1)
    img_asarray = band.ReadAsArray()

    # checando abertura do arquivo
    try:
        img = gdal.Open(img_file, GA_ReadOnly)
        print("Arquivo aberto com sucesso!")
    except:
        print("Erro na abertura do arquivo!")

################################# FUNÇÃO PARA CHECAR A EXTENSÃO ESPACIAL ################################## 

    # transformando coordenadas geográficas nas coordenadas da imagem
    trans = img.GetGeoTransform()

    def world2Pixel(trans, nx, ny):
        ulX = trans[0]
        ulY = trans[3]
        xDist = trans[1]
        yDist = trans[5]
        rtnX = trans[2]
        rtnY = trans[4]
        column = int((nx - ulX) / xDist)
        row = int((ulY - ny) / xDist)

        return (row, column)

################################### ARMAZENANDO O NÚMERO DE PIXEL/CLASSE ###################################

    #criação de vetor que guarda o numero de pixel por classe
    pixel_classe = np.zeros((1, nclass))
    cont1 = 0
    cont2 = 0

    for i in range(nclass):
        while cont2 < nlin_tr and i + 1 == matriz_shape_tr[cont2, 1]:
            cont1 += 1
            cont2 += 1
        pixel_classe[0, i] = cont1
        cont1 = 0

################################## CÁLCULO DO NÚMERO TOTAL DE PIXEL/CLASSE ##################################

    # calculando do valor máximo de pixel armazenado por classe (corresponde a tamanho total de cada classe)

    max_pix = int(pixel_classe.max())

################################## CRIAÇÃO DA MATRIZ COM VALORES ESPECTRAIS #################################

    # criando matriz com valores espectrais da imagem
    matriz_amostras = np.zeros((nband, nclass, max_pix))

    cont2 = 0  # percorre as nlinhas da matriz_shape_tr

    for i in range(img.RasterCount):
        band = img.GetRasterBand(i + 1)
        img_asarray = band.ReadAsArray()
        for j in range(nclass):
            lim_pxcls = int(pixel_classe[0, j])
            for k in range(lim_pxcls):
                pix_xy = world2Pixel(trans, matriz_shape_tr[cont2, 2], matriz_shape_tr[cont2, 3])  #chama as coord. xy da imagem
                valor_espec = img_asarray[pix_xy[0], pix_xy[1]]  #coordenada pix_xy[0] = x e pix_xy[1] = y
                matriz_amostras[i][j][k] = valor_espec
                if cont2 < nlin_tr - 1:
                    cont2 += 1
        cont2 = 0

######################################### CÁLCULO DA MATRIZ DE MÉDIA #########################################

    # matriz das médias
    matriz_media = np.zeros((nband, nclass))
    for i in range(nband):
        for j in range(nclass):
            soma = sum(matriz_amostras[i][j][0:max_pix])
            matriz_media[i][j] = soma / pixel_classe[0, j]

###################################### CÁLCULO DA MATRIZ DE COVARIÂNCIA ######################################

    # cálculo da matriz de covariância por classe
    matriz_cov = np.zeros((nclass, nband, nband))

    for j in range(nclass):
        lim_pxcls = int(pixel_classe[0, j])
        for i in range(nband):
            for k in range(nband):
                matriz_cov[j][k][i] = sum((matriz_amostras[i][j][0:lim_pxcls] - matriz_media[i][j]) * (
                            matriz_amostras[k][j][0:lim_pxcls] - matriz_media[k][j])) / (lim_pxcls - 1)

################################## ARMAZENANDO OS VALORES ESPECTRAIS/BANDA ##################################

    # criando uma matriz para guardar os valores espectrais da imagem por banda
    matriz_img = np.zeros((nband, linhas, colunas))

    for i in range(nband):
        band = img.GetRasterBand(i + 1)
        img_asarray = band.ReadAsArray()
        for j in range(linhas):
            for k in range(colunas):
                matriz_img[i][j][k] = img_asarray[j][k]


##################################### CÁLCULO DA MÁXIMA VEROSSIMILHANÇA #####################################

    # cálculo MaxVer
    matriz_mxv = np.zeros((nclass, linhas, colunas))
    aux1 = np.zeros((nband, 1))
    aux2 = np.zeros((nband, 1))
    band1 = 0

    for i in range(nclass):
        for j in range(linhas):
            for k in range(colunas):
                while (band1 < nband):
                    aux1[band1][0] = matriz_img[band1][j][k]
                    aux2[band1][0] = matriz_media[band1][i]
                    band1 += 1

                band1 = 0

                a = (log(1 / nclass) - 0.5 * log(np.linalg.det(matriz_cov[i][:][:]))) #"np.linalg.det" calcula o determinante de uma matriz            
                b = (-0.5 * transpose(aux1 - aux2)) #"transpose" permuta as dimensões de uma matriz
                c = (inv(matriz_cov[i][:][:])) # "inv" matriz inversa
                d = (aux1 - aux2)
                e = dot(b, c) # função "dot" realiza a multiplicação de matrizes
                f = dot(e, d) 
                matriz_mxv[i][j][k] = a + f

##################################### CRIANDO A MATRIZ DE CLASSIFICAÇÃO #####################################

    # criando matriz de classificação
    map_class = np.zeros((linhas, colunas))
    aux = np.zeros((nclass,1))
    class1 = 0
    max1 = 0
    cont = 0

    for j in range(linhas):
        for k in range(colunas):
            while(class1 < nclass):
                aux[class1][0] = matriz_mxv[class1][j][k]
                class1 += 1         
            max1 = np.max(aux)           
            while (aux[cont][0]!= max1):
                cont += 1           
            map_class[j][k] = cont + 1           
            class1 = 0
            cont = 0
            max1 = 0

##################################### CRIANDO A MATRIZ DE CONFUSÃO #####################################

    # construção da matriz de confusão

    matrix_confu = np.zeros((nclass, nclass))
    cont = 0

    # matriz transposta em realação a do software envi, saída do programa

    for j in range(nclass):
        lim_pxcls = int(pixel_classe[0, j])
        for k in range(lim_pxcls):
            pix_xy = world2Pixel(trans, matriz_shape_tst[cont, 2], matriz_shape_tst[cont, 3])  #chama as coord. xy da imagem
            value_class = int(map_class[pix_xy[0], pix_xy[1]])  #coordenada pix_xy[0] = x e pix_xy[1] = y        
            matrix_confu[j][value_class - 1] = matrix_confu[j][value_class - 1] + 1
            if cont < nlin_tr - 1:
                cont += 1

    # matriz de acordo com a saída do software envi.

    matrix_confu_inv = np.zeros((nclass, nclass))

    for j in range(nclass):
        for k in range(nclass):
            matrix_confu_inv[j][k] = matrix_confu[k][j]

##################################### CALCULANDO O ÍNDICE DE KAPPA E ACURÁCIAS #####################################

    # construção do índice kappa
    soma = 0  # valor total de pixel da amostras de teste
    cont1 = 0

    vet_col = np.zeros((1, nclass))
    vet_lin = np.zeros((1, nclass))
    sum_diag = 0
    diag = np.zeros((1, nclass))

    for k in range(nclass):
        soma = soma + pixel_classe[0, k]

    # somando as linhas da matriz
    for i in range(nclass):
        for j in range(nclass):
            cont1 = cont1 + matrix_confu[i][j]
            if i == j:
                sum_diag = sum_diag + matrix_confu[i][j]
                diag[0][i] = matrix_confu[i][j]

        vet_lin[0, i] = cont1
        cont1 = 0

    # somando as colunas da matriz
    for i in range(nclass):
        for j in range(nclass):
            cont1 = cont1 + matrix_confu[j][i]
        vet_col[0, i] = cont1
        cont1 = 0

    # calculando a acurácia global
    po = sum_diag / soma
    ac_global = po * 100  # acurácia global

    pc = (sum(vet_col * vet_lin)) / (soma ** 2)

    kappa = (po - pc) / (1 - pc)

    # calculando a acurácia do usuário
    ac_usuario = (diag / vet_col) * 100

    # calculando a acurácia do produtor
    ac_produtor = (diag / vet_lin) * 100
    
###################################### SALVANDO O RESULTADO EM TXT ######################################

    # salvando a matriz em arquivo txt
    num_cls = np.zeros((1, nclass))
    for i in range(nclass):
        num_cls[0][i] = i + 1

    with open('C:/Users/Maria/_trabalho_final_intro/saida_txt/{}.txt'.format(nome_arq[num_arq]), "w") as out_file:    
    
        for i in range(len(num_cls)):
            out_string = " "
            out_file.write(out_string)
            out_string += str(num_cls[i]) 
            out_file.write(out_string)
        out_string = ("\n")
        out_file.write(out_string)
        for i in range(len(matrix_confu_inv)):
            out_string = ""
            out_string += str(i + 1)
            out_string += str(matrix_confu_inv[i])    
            out_string += "\n"
            out_file.write(out_string)
        out_string = ("\n")
        out_file.write(out_string)
        out_string = ("Kappa: ")
        out_file.write(out_string)
        out_string = str(kappa)   
        out_file.write(out_string)
        out_string = ("\n")
        out_file.write(out_string)
        out_string = ("Acurácia Global: ")
        out_file.write(out_string)
        out_string = str(ac_global)
        out_file.write(out_string)
        out_string = ("\n")
        out_file.write(out_string)
        out_string = ("Acurácia do usuário: ")
        out_file.write(out_string)    
        for i in range(len(ac_usuario)):
            out_string = ""
            out_string += str(ac_usuario[i])
            out_string += "\n"
            out_file.write(out_string)
        out_string = ("Acurácia do produtor: ")
        out_file.write(out_string)
        for i in range(len(ac_produtor)):
            out_string = ""
            out_string += str(ac_produtor[i])
            out_string += "\n"
            out_file.write(out_string)
            
    num_arq += 1

############################################# LIBERANDO MEMÓRIA #############################################

#fenchando as imagens para liberar memória

img_file = None