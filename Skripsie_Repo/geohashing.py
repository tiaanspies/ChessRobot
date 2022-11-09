import numpy as np
import cv2 as cv
import trackingFunctions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

P = np.array([3.0661702,2.33220631, 433.95037577])
Q = np.array([-37.64655527,-50.63663523,415.40080437])
R = np.array([-93.63437281,-13.99496436,419.90303379])

#Z = [0,-50,20] (x,y,z)

X_PR= R[0]-P[0] 
Y_PR = R[1]-P[1]
Z_PR = R[2]-P[2]
print ("Debug")
print(X_PR,Y_PR,Z_PR)
#find two vectors in the plane 
PQ = Q-P

PR = R-P # u vector
print (PR)

#print (PR)
#find a vector normal to these two vectors (using cross product)
n = np.cross(PR,PQ)
a,b,c = n 
d = np.dot(n, P)
d1 = 264.394
d2 = 80
#print (d1)
### I know have my parametres of my plane
# *AX=B
A = np.array([[a,b,c],[X_PR,Y_PR,Z_PR],[2*Q[0]-2*P[0],2*Q[1]-2*P[1],2*Q[2]-2*P[2]]])
#print (A)
B = np.array([[d],[X_PR*Q[0]+Y_PR*Q[1]+Z_PR*Q[2]],[Q[0]**2-P[0]**2+Q[1]**2-P[1]**2+Q[2]**2-P[2]**2+d1**2-d2**2]])
#print (B)
x = np.linalg.lstsq(A,B)[0]
x = np.reshape(x,3)
print (x)

X = [P[0],Q[0],R[0],x[0]]
Y = [P[1],Q[1],R[1],x[1]]
Z = [P[2],Q[2],R[2],x[2]]


xx, yy = np.meshgrid(range(50), range(50))
#z = (-n[0] * xx - n[1] * yy - d) * 1. /n[2]
zz = (d- a*xx - b*yy)/c 



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,Z,c='r', marker='o')
ax.text(P[0],P[1],P[2],"P")
ax.text(Q[0],Q[1],Q[2],"Q")
ax.text(R[0],R[1],R[2],"R")
ax.text(x[0],x[1],x[2],"M")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
#ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)
#ax.text("Q",(Q[0],Q[1],Q[2]))
#ax.text("R",(R[0],R[1],R[2]))
#plt.annotate("M",x)
ax.plot_surface(xx, yy, zz, alpha=0.2)
plt.show()
