import numpy as np
import cv2
from matplotlib import pyplot as plt

def filtrado(x,y,X1, Y1):#acota la region de busqueda de la bacteria senialada con el punto
	indices = []
	for i in range(0, len(X1)):
		if (X1[i] <= x + 18) and (Y1[i] <= y + 18):
			if (X1[i] >= x - 18) and (Y1[i] >= y - 18):
				indices.append(i)
	return indices

def encontrar(x, y, indices, cont):#encuentra el indice de la bacteria señalada con el punto (se supone XD)
	minimo = 999.0
	encontrado = 0
	distancias = []
	for i in range(0, len(indices)):
		for j in range(0, len(cont[indices[i]])):
			x1 = cont[indices[i]][j][0][0]
			y1 = cont[indices[i]][j][0][1]
			tmp = np.sqrt(((x-x1)**2)+((y-y1)**2))
			distancias.append(np.sqrt(((x-x1)**2)+((y-y1)**2)))

		tmp = min(distancias)
		
		if tmp < minimo:
			minimo = tmp
			encontrado = i
		
		del distancias[:]

	return indices[encontrado]


cap = cv2.VideoCapture("vidYES5.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')#para guardar videos
frame_width = int(cap.get(3))# //
frame_height = int(cap.get(4))# //
out = cv2.VideoWriter('seguimiento5v3.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 3, (frame_width,frame_height))# //
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

lk_params = dict( winSize  = (18,18),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 18, 0.001))
 
#Capturamos una imagen y la convertimos de RGB -> HSV
_, imagen = cap.read()
frame = imagen.copy()

kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((3,3),np.uint8)
kernel3 = np.ones((6,6),np.uint8)
kernel4 = np.ones((7,7),np.uint8)
kernel5 = np.ones((2,2),np.uint8)
kernel6 = np.ones((4,4),np.uint8)

gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

ret, thresh2= cv2.threshold(gris,100,255,cv2.THRESH_BINARY)#parametros iteracion anterior(video 3a):133,255
#ima2= cv2.erode(thresh2,kernel5,iterations = 1)

(img, contornos, jerarquia) = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

X = []
Y = []
Ma = []
mA = []
Angle = []
contornos2 = []


for i in range(0, len(contornos)):#se guardan los valores de las elipses que mas se asemejan a las bacterias
	if(len(contornos[i])>4 and jerarquia[0][i][2]==-1 and jerarquia[0][i][3]!=-1):
		(x,y),(MA,ma),angle = cv2.fitEllipse(contornos[i])
		#x, y son el centro, MA es el ancho y ma el alto 
		if ((MA < 65) and (MA >8) and (ma <65) and (ma > 8)):
			contornos2.append(contornos[i])
			ellipse = cv2.fitEllipse(contornos[i])
			X.append(x)
			Y.append(y)
			Ma.append(MA)
			mA.append(ma)
			Angle.append(angle)

momentos=[]
punto_a_encontrar = [0,5]#5

for h in range(0,len(punto_a_encontrar)):
    	momentos.append(cv2.moments(contornos2[punto_a_encontrar[h]]))

frame_anterior = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

punto_elegido=[]

for k in range(0,len(punto_a_encontrar)):
	punto_elegido.append(np.array([[[X[punto_a_encontrar[k]], Y[punto_a_encontrar[k]]]]],np.float32))

count = 0
nframes = 0

while(True):
    ret, frame = cap.read()

    if ret==True:
        nframes = nframes + 1
        imagen = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
		#cv2.putText(imagen,'Divitions: '+div_text,(40, 50), font, 1,(255,255,255),1,cv2.LINE_AA)
		#cv2.putText(imagen,'Time of the last divition: '+seg_text,(40, 90), font, 1,(255,255,255),1,cv2.LINE_AA)
		# Convert BGR to HSV
        bitwise = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        gris1 = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

		#se aplica el metodo de Lucas Kanade
        for j in range(0,len(punto_elegido)):
            punto_elegido[j], st, err = cv2.calcOpticalFlowPyrLK(frame_anterior, gris1, punto_elegido[j],None, **lk_params)
		#print("punto elegido: ", punto_elegido[0][0])

		#Se guarda el frame de la iteracion anterior del bucle
        frame_anterior = gris1.copy()

        kernel = np.ones((5,5),np.uint8)
        kernel2 = np.ones((3,3),np.uint8)
        kernel3 = np.ones((6,6),np.uint8)
        kernel4 = np.ones((7,7),np.uint8)
        kernel5 = np.ones((2,2),np.uint8)
        kernel6 = np.ones((4,4),np.uint8)


        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imagen2=cv2.bitwise_not(gris)

        ret, thresh2= cv2.threshold(gris,100,255,cv2.THRESH_BINARY) #parametros iteracion anterior(video 3a): 135,255
        #ima2= cv2.erode(thresh2,kernel5,iterations = 1)

            

        (img, contornos, jerarquia) = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        X = []
        Y = []
        Ma = []
        mA = []
        Angle = []
        contornos2 = []


        for i in range(0, len(contornos)):#se guardan los valores de las elipses que mas se asemejan a las bacterias
            if(len(contornos[i])>4 and jerarquia[0][i][2]==-1 and jerarquia[0][i][3]!=-1):
                (x,y),(MA,ma),angle = cv2.fitEllipse(contornos[i])
				#x, y son el centro, MA es el ancho y ma el alto 
                if ((MA < 65) and (MA >8) and (ma <65) and (ma > 8)):
                    contornos2.append(contornos[i])
                    ellipse = cv2.fitEllipse(contornos[i])
                    X.append(x)
                    Y.append(y)
                    Ma.append(MA)
                    mA.append(ma)
                    Angle.append(angle)

        c = 0

        puntos=[]

        for o in range(0,len(punto_elegido)):
            puntos.append(filtrado(punto_elegido[o][0][0][0], punto_elegido[o][0][0][1],X, Y))

        for u in range(0,len(puntos)):
            if(len(puntos[u]) >0):
                punto = encontrar(punto_elegido[u][0][0][0], punto_elegido[u][0][0][1], puntos[u], contornos2)
                punto_elegido2 = np.array([[[X[punto], Y[punto]]]],np.float32)
                largo2 = mA[punto]

                for i in range(0, len(puntos[u])):
                    c = puntos[u][i]
                    ellipse = (X[c], Y[c]), (Ma[c], mA[c]), Angle[c]
                    im = cv2.ellipse(imagen,ellipse,(255,0,0),1, cv2.LINE_AA)

                for i in punto_elegido2:
                    ellipse = (X[punto], Y[punto]), (Ma[punto], mA[punto]), Angle[punto]
                    im = cv2.ellipse(imagen,ellipse,(0,255,0),1, cv2.LINE_AA)

				#for i in range(0, len(X)):#se aplica un amplificador para agrandar los margenes de los bordes
				#	Ma[i] = Ma[i] * 1.2
				#	mA[i] = mA[i] * 1.2

                for i in range(0, len(X)):#se dibujan las elipses con bordes amplificados
                    ellipse = (X[i], Y[i]), (Ma[i], mA[i]), Angle[i]
                    im = cv2.ellipse(bitwise,ellipse,(0,0,255),1, cv2.LINE_AA)

                ellipse = (X[0], Y[0]), (Ma[0], mA[0]), Angle[0]
                im = cv2.ellipse(bitwise,ellipse,(255,0,0),-1, cv2.LINE_AA)
                cv2.drawContours(frame, contornos2, -1, (0,255,0),1)

		#cv2.imshow('Frame',frame)
		#cv2.imshow('bitwise',bitwise)
        cv2.imshow('imagen', imagen)
		#cv2.imwrite("imagen%d.jpeg" %count, imagen)
		#div_text = str(count)
        count = count +1
        out.write(imagen) # escribe el frame actual(para el video)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        largo1 = largo2
        largo2 = 0
        
    else:
    	break