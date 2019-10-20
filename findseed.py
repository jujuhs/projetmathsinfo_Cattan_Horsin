import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return np.exp(-x**2-y**2)


def find_seed(f,c=0.0,x=0.0,eps=2**(-26)):
    if not f(x,1.0)<=c<=f(x,0.0) and not f(x,0.0)<=c<=f(x,1.0):
        return None
    t=0.5
    h=10**(-8)
    while abs(f(x,t)-c)>eps:  ##dichotomie des familles mais Newton ne fonctionnait pas...
        fprime=(f(x,t+h)-f(x,t))/h
        t=t-(f(x,t)/fprime)
    return t 


#print(find_seed(f,float(c)))

def simple_contour(f,c=0.0,delta=0.01):
    x=[]
    y=[]
    for i in np.arange(0,1,delta):
        if find_seed(f,c,i) != None :
            y+=[find_seed(f,c,i)]
            x+=[i]
    return [x,y]

c=input("Donnez la valeur du réel c >> ")    
print(simple_contour(f,float(c)))
plt.plot(simple_contour(f,float(c))[0],simple_contour(f,float(c))[1])
plt.grid()
plt.show()

