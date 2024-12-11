#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:30:43 2024

@author: nathanielcutler
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### inital condition

nx = 30 
ny = 40

dx = 2 #meters 
dy = 2 #meters 

x = np.arange (0,nx*dx,dx) ## creates 1d array of x positions 
y = np.arange (0,ny*dy,dy) ## creates 1d array of y positions 

X, Y = np.meshgrid(x,y, indexing = "ij") ## creates  2d coordinate system for plotting 

"""
cappital letters = 2d array 
lowercase = 1d array 
"""

Dx = 0.02 #m2/year 
Dy = 0.02

dt = 5 #years 

## lets do topogrtaphy 

Z = np.random.random ((nx,ny))*100 ## 2d because this is capitalized 

z = Z.flatten() ## gives a 1d flattened array, flattened by rows (x rows y collumns)


## stability check 

sx = dt * Dx / dx**2

sy = dt * Dy / dy**2

import sys
if sx>0.5: 
    print ("x is unstable")
    sys.exit()
elif sy > 0.5: 
    print("y is unstable")
    sys.exit()
    


### A matrix 

A = np.zeros ((nx*ny, nx*ny))

for i in range(nx): 
    for k in range(ny):
        ik = i*ny + k
        ### ---boundary conditions ---
        if i == 0: 
            A[ik,ik] = 1 ## no change 
        elif i == (nx-1): 
            A[ik,ik]=1
        elif k == 0: 
            A[ik,ik]=1
        elif k == (ny-1):
            A[ik,ik]=1
        else:
            ##--- matrix coefficient time---
            A[ik,ik] = 1-2*sx -2*sy 
            A[ik,(i+1)*ny + k] = sx 
            A[ik,(i-1)*ny + k] = sx 
            A[ik, i*ny +k + 1] = sy
            A[ik, i*ny +k - 1] = sy

#print (A)

##----plot initial consitions----
## method 1 - surface plot

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.plot_surface(X,Y,Z)
ax.set_title('inital conditions')
ax.set_xlabel('x')
ax.set_zlabel('z')
ax.set_ylabel('y')
##method 2 - pcolormesh 

fig2, ax2 = plt.subplots (1,1)
c = ax2.pcolormesh(X,Y,Z)
ax2.set_title('inital conditions')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
colorbar = fig2.colorbar (c,ax=ax2)
colorbar.set_label("elevation")

##running time 
time=0 
totaltime = 1000 #years 
while time <= totaltime: 
    new_z = np.dot (A,z)
    z[:] = new_z*1
    time += dt



##----plot final consitions----
## method 1 - surface plot
Z = z.reshape (X.shape)
fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.plot_surface(X,Y,Z)
ax.set_title('Final Conditions')
ax.set_xlabel('x')
ax.set_zlabel('z')
ax.set_ylabel('y')


##method 2 - pcolormesh 

fig2, ax2 = plt.subplots (1,1)
c = ax2.pcolormesh(X,Y,Z)
ax2.set_title('final conditions ')
colorbar = fig2.colorbar (c,ax=ax2)
colorbar.set_label("elevation")




