import numpy as np
import matplotlib.pyplot as plt


def f(x,y):
    return np.exp(-x**2-y**2)


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


#print(find_seed(f,float(c)))

def distance(x,y):
    return np.sqrt(((x[0]-y[0])**2)+((x[1]-y[1])**2))

def simple_contour(f,c=0.0,delta=0.01):
    x=[0]
    y=[find_seed(f,c,0)]
    Stock=[]
    distance=[]
    while x
        for j in np.arange(i,i+delta,delta/1000):
           if find_seed(f,c,j) != None :
               Stock+=[j]
        for a in Stock :
            X=[x[-1],c]
            Y=[find_seed(f,c,a),c]
            distance+=[distance(X,Y)]
        e=distance.index(min(distance))
        y+=[find_seed(f,c,Stock[e])]
        x+=[Stock[e]]
    return [x,y]

c=input("Donnez la valeur du réel c >> ")    
print(simple_contour(f,float(c)))
plt.plot(simple_contour(f,float(c))[0],simple_contour(f,float(c))[1])
plt.grid()
plt.title(f"Courbe de niveau pour c={float(c)}")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

