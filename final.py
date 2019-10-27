import matplotlib.pyplot as plt  
import autograd
from autograd import numpy as np




def distanceeucl(x, y):
    return np.sqrt(((x[0]-y[0])**2)+((x[1]-y[1])**2))

def normalisation(vecteur,normev): # l'idée est de rendre le vecteur unitaire puis de le multiplier par la norme souhaitée
    norme = distanceeucl(vecteur,(0,0))
    x = vecteur[0]
    y = vecteur[1]
    return (x*normev/norme, y*normev/norme)

def grad(f, x, y):
    g = autograd.grad
    return np.r_[g(f, 0)(x, y), g(f, 1)(x, y)]







def f(x,y):
    return 2*(np.exp(-x**2-y**2)-np.exp(-(x-1)**2-(y-1)**2))

def h(x,y):
    return x**2 + (y-0.4)**2












def find_seedx(f, c=0.0, x=0.0, y1 = 0.0, y2 = 1.0, eps = 2**(-26)): # cette fonction permet à x fixé, de chercher un y
    if not f(x,y1)<= c <= f(x,y2) and not f(x,y2)<= c <= f(x, y1):  # compris entre y1 et y2 tel que f(x,y)=c
            return None  
    a = y1
    b = y2
    t = (a+b)/2 
    while abs(f(x,t)- c) > eps :  
        if (f(x, a) - c)*(f(x,t) - c) >= 0:
            a = t
        else :
            b = t
        t = (a+b)/2
    return t 












def find_seedy(f, c=0.0, y=0.0, x1 = 0.0, x2 = 1.0, eps = 2**(-26)): # cette fonction permet à y fixé, de chercher un x
    if not f(x1, y)<= c <= f(x2, y) and not f(x2, y)<= c <= f(x1, y): # compris entre x1 et x2 
            return None  
    a = x1
    b = x2
    t = (a+b)/2
    while abs(f(t, y)-c) > eps :  
        if (f(a, y)-c)*(f(t, y)-c) >= 0:
            a = t
        else :
            b = t
        t = (a+b)/2
    return t 















def simple_contourmodif(f,c=0.0,delta=0.01, x1 = 0.0, x2 = 1.0, y1 = 0.0, y2 = 1.0):
    coté_X = [x1, x2]
    coté_Y = [y1, y2]
    test_coté = False
    for i in coté_X: #on réalise une recherche de premier point sur les droite d'équation x=x1 et x=x2 
        a = find_seedx(f, c, i, y1, y2)
        if a != None:
            X = [i]
            Y = [a]
            test_coté = True
            
    for i in coté_Y: #on réalise une recherche de premier point sur les droites d'équation y=y1 et y=y2
        a = find_seedy(f, c, i, x1, x2)
        if a != None:
            Y = [i]
            X = [a]
            test_coté = True
            
    if test_coté == False: #si aucun point n'a été trouvé sur les bords, on retourne des listes vides. 
        return [[],[]]
    
    
    point_précédent = [X[-1],Y[-1]]
    gradient = grad(f, point_précédent[0], point_précédent[1]) #on calcule le gradient au premier point trouvé
    tangente = [-gradient[1], gradient[0]] #le gradient dirige la normale à la tangente.
    tangente_normalisée = normalisation(tangente, delta) #la norme du vecteur tangent est delta
    
    if x1 <= (point_précédent[0] + tangente_normalisée[0]) <= x2 and  y1 <= point_précédent[1] + tangente_normalisée[1] <= y2:  #on vérifie que le point choisi en second partira bien vers la droite (x>=0)
        deuxième_point = (point_précédent[0] + tangente_normalisée[0],
                          point_précédent[1] + tangente_normalisée[1]
                         )
    else : 
        deuxième_point = (point_précédent[0] - tangente_normalisée[0],
                          point_précédent[1] - tangente_normalisée[1]
                         )
         
    nouveau_point = np.array(deuxième_point)
    def F(x, y): #on définit le système à 2 équations
        return np.array([f(x,y) - c,(x - point_précédent[0])**2 + (y - point_précédent[1])**2 - delta**2])
        
    def Jacob(F, x, y):  #on définit la jacobienne.
        j = autograd.jacobian
        return np.c_[j(F,0)(x, y),j(F,1)(x, y)]
    
    A = F( nouveau_point[0],  nouveau_point[1])   #matrice associéé au système 2x2
    
    while distanceeucl(A,[0,0]) >= 2**(-10) :
        nouveau_point =  nouveau_point - np.linalg.inv(Jacob(F,  nouveau_point[0], nouveau_point[1])).dot(np.array(A))
        A = F( nouveau_point[0],  nouveau_point[1])
    X += [ nouveau_point[0]]
    Y += [ nouveau_point[1]]
    
    
    while X[-1] <= x2 and Y[-1] <= y2  and X[-1] >= x1 and Y[-1] >= y1 :
        distance = []
        point_précédent = [X[-1], 
                           Y[-1]
                          ]
    
        gradient = grad(f, point_précédent[0], point_précédent[1])
        tangente=[-gradient[1],gradient[0]]
        tangente_normalisée = normalisation(tangente,delta)
        point1 = [point_précédent[0] + tangente_normalisée[0],
                  point_précédent[1] + tangente_normalisée[1]
                 ]
        
        point2 = [point_précédent[0] - tangente_normalisée[0],
                  point_précédent[1] - tangente_normalisée[1]]
        liste_point = [point1, point2]
        
        for i in liste_point:            
            distance += [distanceeucl((X[-2], Y[-2]), i)]
            
        e = distance.index(max(distance))       
        nouveau_point = np.array(liste_point[e])
        
        def F(x, y): #on définit le système à 2 équations
            return np.array([f(x,y) - c,(x - point_précédent[0])**2 + (y - point_précédent[1])**2 - delta**2])
        
        def Jacob(F, x, y):  #on définit la jacobienne.
            j = autograd.jacobian
            return np.c_[j(F,0)(x, y),j(F,1)(x, y)]
        
        A = F(nouveau_point[0], nouveau_point[1])   
        
        while distanceeucl(A,[0,0]) >= 2**(-10):            
            A = F(nouveau_point[0],nouveau_point[1])
            nouveau_point = nouveau_point - np.linalg.inv(Jacob(F, nouveau_point[0], nouveau_point[1])).dot(np.array(A))
        X += [nouveau_point[0]]
        Y += [nouveau_point[1]]
    return [X, Y]



















def contour(f,c, xc = [0.0,1.0], yc = [0.0,1.0], delta = 0.01): #la fonction prend en argument le quadrillage défini par les listes xc et yc.
    X_tot = []
    Y_tot = []
    for i in range(len(yc) - 1):
        for j in range(len(xc)-1): #dans chaque petit rectangle, on applique simple_contourmodif.
            xg = xc[j]
            xd = xc[j+1]
            yg = yc[i]
            yd = yc[i+1]
            R = simple_contourmodif(f, c, delta , xg, xd, yg, yd)
            X_tot.append(R[0])
            Y_tot.append(R[1])
    return [X_tot, Y_tot]

    


















xc = np.linspace(-2,3,10) #on définit le quadrillage relatif à la fonction exemple f
yc = np.linspace(-1,2,5)
valeur = np.arange(-1.5, 2, 0.5) #on définit les différentes valeur de c.
v = ["b","c","g","r","m","b","y"]
for i in range(len(valeur)):
    data = contour(f,float(valeur[i]), xc, yc)
    for x,y in zip(data[0],data[1]):
        plt.plot(x, y, color = v[i])
plt.grid()
plt.title(f"Courbe de niveau pour c dans {valeur}")
plt.xlabel("x")
plt.ylabel("y")
plt.show()