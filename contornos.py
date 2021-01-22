import numpy as np
import cv2
from matplotlib import pyplot as plt
import statistics
import math


def es_positivo(numero):
    aux=abs(numero)
    if(numero*aux<0):
        return False
    else:
        return True


def es_elipse(contorno):
    status = True
    # aux=cv2.approxPolyDP(contorno,0.02*cv2.arcLength(contorno,True),True)
    # print(len(aux))
    # if(len(aux)==7 or len(aux)==8):
    temp = cv2.approxPolyDP(contorno, 0.01*cv2.arcLength(contorno, True), True)
    for j in range(0,len(temp)-2):
        aux1=temp[j][0]
        aux2=temp[j+1][0]
        aux3=temp[j+2][0]
        #aux4=temp[j+3][0]
        m1=(aux3[1]-aux2[1])/(aux3[0]-aux2[0])
        m2=(aux2[1]-aux1[1])/(aux2[0]-aux1[0])
        tangente=(m2-m1)/(1+(m2*m1))
        angulo=math.atan(tangente)
        angulo=(angulo)
        angulo=math.degrees(angulo)
        global cont
        if(str(angulo) != 'nan'):
            if(abs(angulo)>=86):
                status=False
                return status
    return status

imagen = cv2.imread('new_frame_5_0v2.png', 0)
imagen8 = cv2.imread('new_frame_5_0v2.png', 1)
imagen9 = cv2.imread('new_frame_5_0v2.png', 1)
imagen10 = cv2.imread('blanco.jpg', 1)

kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((3, 3), np.uint8)
kernel3 = np.ones((6, 6), np.uint8)
kernel4 = np.ones((7, 7), np.uint8)
kernel5 = np.ones((2, 2), np.uint8)
kernel6 = np.ones((4, 4), np.uint8)

ret, thresh = cv2.threshold(imagen, 100, 255, cv2.THRESH_BINARY)

canny = cv2.Canny(thresh, 200, 255)

(img, contornos, jerarquia) = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

X = []
Y = []
Ma = []
mA = []
Angle = []
contornos2 = []
cont = 0 


# se guardan los valores de las elipses que mas se asemejan a las bacterias
for i in range(0, len(contornos)):
    if(len(contornos[i]) > 4 and jerarquia[0][i][2] == -1 and jerarquia[0][i][3] != -1 and es_elipse(contornos[i]) == True):
        (x, y), (MA, ma), angle = cv2.fitEllipse(contornos[i])
         
        # x, y son el centro, MA es el ancho y ma el alto
        if ((MA < 65) and (MA > 5) and (ma < 65) and (ma > 5)): #and angle!=0.0):
            contornos2.append(contornos[i])
            ellipse = cv2.fitEllipse(contornos[i])
            X.append(x)
            Y.append(y)
            Ma.append(MA)
            mA.append(ma)
            Angle.append(angle)
    
cv2.drawContours(imagen10, contornos2, -1, (255, 0, 0), 1)
print("cantidad de contornos: {}".format(len(contornos2)))
#print("ancho: {}".format(Ma[0]))
#print("largo: {} ".format(mA[0]))
cv2.drawContours(imagen8, contornos2[0], -1, (255, 0, 0), 1)  # azul
cv2.drawContours(imagen8, contornos2[1], -1, (0,255,0),1)#verde
cv2.drawContours(imagen8, contornos2[2], -1, (0,0,255),1)#rojo
cv2.drawContours(imagen8, contornos2[27], -1, (100,100,255),1)#rosado
cv2.drawContours(imagen8, contornos2[28], -1, (0, 255, 255), 1)  # amarillo
cv2.drawContours(imagen8, contornos2[29], -1, (255,255,0),1)#celeste
cv2.drawContours(imagen8, contornos2[30], -1, (255,0,255),1)#morado
cv2.drawContours(imagen8, contornos2[31], -1, (0,10,100),1)
print("Ma(ancho) "+ str(Ma[0])+ " mA(largo) "+ str(mA[0]) + " Angle "+ str(Angle[0])+" azul")
print("Ma(ancho) "+ str(Ma[1])+ " mA(largo) "+ str(mA[1]) + " Angle "+ str(Angle[1])+" verde")
print("Ma(ancho) "+ str(Ma[2])+ " mA(largo) "+ str(mA[2]) + " Angle "+ str(Angle[2])+" rojo")
print("Ma(ancho) "+ str(Ma[28])+ " mA(largo) "+ str(mA[28]) + " Angle "+ str(Angle[28])+" amarillo")
print("Ma(ancho) "+ str(Ma[27])+ " mA(largo) "+ str(mA[27]) + " Angle "+ str(Angle[27])+" rosado")

#print(statistics.stdev(Ma) )
#print(statistics.stdev(mA) )
cv2.imshow('imagen10', imagen10)
cv2.imshow('imagen8', imagen8)
cv2.imshow('imagen9', imagen9)
# cv2.imshow('imagen_con_contornos',imagen8)

cv2.waitKey(0)
