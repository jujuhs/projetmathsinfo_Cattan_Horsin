import numpy as np
import matplotlib.pyplot as plt


def f(x,y):
    return 2*(np.exp(-x**2-y**2)-np.exp(-(x-1)**2-(y-1)**2))


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


def simple_contour(f,c=0.0,delta=0.01):
    x=[0]
    y=[find_seed(f,c,0)]   
    presence=True 
    while presence==True and x[-1]<1-delta:
        stock=[]
        distance=[]
        for j in np.arange(x[-1],x[-1]+delta,delta/1000):
            if find_seed(f,c,j) != None :
                stock+=[[j,find_seed(f,c,j)]]
        avant=[x[-1],y[-1]]
        for a in stock :
            apres=a
            distance+=[distanceeucl(avant, apres)]       
        L=[abs(distance[i]-delta) for i in range(len(distance))]
        e=L.index(min(L))
        if distanceeucl(avant,stock[e]) < (delta/10) :
            presence=False
        y+=[stock[e][1]]
        x+=[stock[e][0]]
        print(stock[e][0])
    return [x,y]

c=input("Donnez la valeur du réel c >> ")    
data=simple_contour(f,float(c))
#print(simple_contour(f,float(c)))
plt.plot(data[0],data[1])
plt.grid()
plt.title(f"Courbe de niveau pour c={float(c)}")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

