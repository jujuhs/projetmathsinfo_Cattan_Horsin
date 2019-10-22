import numpy as np  
import matplotlib.pypylot as plt  

def find_seed(f,c=0.0,x=0.0,eps=2**(-26)):
    if not f(x,1.0)<= c <= f(x,0.0) and not f(x,0)<=c<=f(x,1.0):
            return None  
    a=0
    b=1
    t=0.5
    while abs(f(x,t)-c)>eps:  ##dichotomie des familles mais Newton ne fonctionnait pas...
        if (f(x,a)-c)*(f(x,t)-c)>=0:
            a=t
        else :
            b=t
        t=(a+b)/2
    return t 

def distanceeucl(x,y):
    return np.sqrt(((x[0]-y[0])**2)+((x[1]-y[1])**2))

def derivee_prem_coord(f,x,y,h=10**-8):
    return (f(x+h,y)-f(x,y))/h

def derivee_deux_coord(f,x,y,h=10**-8):
    return (f(x,y+h)-f(x,y))/h

def simple_contour(f,c=0.0,delta=0.01):
    x=[0]
    y=[find_seed(f,c,0.0)]
    while  #condition d'arrêt à trouver ie intersection d'un bord : 

        vect_tang=[-deriv_deux_coord(f,x[-1],y[-1]),deriv_prem_coord(f,x[-1],y[-1])]  ##vecteur tangent à la ligne de niveau au dernier point trouvé. 
        norme_vect_tang = distanceeucl(vect_tang,[0,0])
        vect_tang=(delta/norme_vect_tang)*np.array(vect_tang)
        apres1=[]
    
