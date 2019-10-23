import matplotlib.pyplot as plt  
import autograd
from autograd import numpy as np



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

def f(x,y):
    return 2*(np.exp(-x**2-y**2)-np.exp(-(x-1)**2-(y-1)**2))

def g(x,y):
    return x**2+(y-0.4)**2

def distanceeucl(x,y):
    return np.sqrt(((x[0]-y[0])**2)+((x[1]-y[1])**2))

def normalisation(vecteur,normev):
    norme=distanceeucl(vecteur,(0,0))
    x=vecteur[0]
    y=vecteur[1]
    return (x*normev/norme,y*normev/norme)


#def derivee_prem_coord(f,x,y,h=10**-4):
    #return (f(x+h,y)-f(x,y))/h

#def derivee_deux_coord(f,x,y,h=10**-4):
    #return (f(x,y+h)-f(x,y))/h


def grad(f,x,y):
    g=autograd.grad
    return np.r_[g(f,0)(x,y),g(f,1)(x,y)]


def simple_contour(f,c=0.0,delta=0.01):
    x=[0.0]
    y=[find_seed(f,c,0.0)]          
    avant=[x[-1],y[-1]]
    gradient=grad(f,avant[0],avant[1])
    tang=[-gradient[1],gradient[0]]
    v=normalisation(tang,delta)
    p1=(avant[0]+v[0],avant[1]+v[1])
    x+=[p1[0]]
    y+=[p1[1]]
    while x[-1]<1-delta and y[-1]>0+delta and y[-1]<1-delta:
        distance=[]
        avant=[x[-1],y[-1]]
        gradient=grad(f,avant[0],avant[1])
        tang=[-gradient[1],gradient[0]]
        v=normalisation(tang,delta)
        p1=[avant[0]+v[0],avant[1]+v[1]]
        p2=[avant[0]-v[0],avant[1]-v[1]]
        p=[p1,p2]
        for i in p:
            distance+=[distanceeucl((x[-2],y[-2]),i)]
        e=distance.index(max(distance))
        
        def F(x,y):
            return np.array([f(x,y)-c,(x-avant[0])**2+(y-avant[1])**2-delta**2])
        
        def Jacob(F,x,y):                #on définit la jacobienne.
            j = autograd.jacobian
            return np.c_[j(F,0)(x,y),j(F,1)(x,y)]
       
        X=np.array(p[e])
        A=F(X[0],X[1])   ##Il s'agit du point intermédiaire permettant d'initialiser Newton.
        while distanceeucl(A,[0,0]) >= 2**(-10) :
            A=F(X[0],X[1])
            X=X-np.linalg.inv(Jacob(F,X[0],X[1])).dot(np.array(A))
        x+=[X[0]]
        y+=[X[1]]
    return [x,y]

c=input("Donnez la valeur du réel c >> ")    
data=simple_contour(f,float(c))
print(simple_contour(f,float(c)))
plt.plot(data[0],data[1])
plt.grid()
plt.title(f"Courbe de niveau pour c={float(c)}")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
