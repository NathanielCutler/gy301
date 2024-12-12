#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  11 12:28:06 2024

@author: sschanz
"""

import numpy as np
import matplotlib.pyplot as plt


### -------- INITIAL CONDITIONS -------------

# import .asc data
ascii_grid = np.loadtxt("gy301_3dprint.asc", dtype = 'float', skiprows=6)

ascii_headers = np.loadtxt("gy301_3dprint.asc", max_rows = 6, dtype = 'str')
n_long = ascii_headers[0,1].astype(int)
n_lat = ascii_headers[1,1].astype(int)
dxy = ascii_headers[4,1].astype(float)
xllcorner = ascii_headers[2,1].astype(float)
yllcorner = ascii_headers[3,1].astype(float)

## ---- GRID FORMATION
x = np.arange(0, dxy*n_lat, dxy) + xllcorner # array of x values
y = np.arange(0, dxy*n_long, dxy) + yllcorner # array of z values
LAT, LONG = np.meshgrid(x, y, indexing='ij') # this sets up a plotting grid
nodes = n_long*n_lat

## ---- TIME STEP
dt = 5 # year

dx = dxy #meters 
dy = dxy #meters 

## ---- DIFFUSION
D = 0.004 # m2/year

Dx = D
Dy = D

## ---- INITIAL ELEVATION
##for i in range(n_lat):
    ##ascii_grid[0:i] = ascii_grid [1:i]
ascii_grid [ascii_grid <0] = np.nan## gets rid of holes in the elevation data 
elv_flat = ascii_grid.flatten()



### stability conditions 

sx = dt * Dx / dx**2

sy = dt * Dy / dy**2

import sys
if sx>0.5: 
    print ("x is unstable")
    sys.exit()
elif sy > 0.5: 
    print("y is unstable")
    sys.exit()
    


### --------- PLOT INITIAL CONDITIONS -------------

fig, ax = plt.subplots(1,1) # use this method
elv_matrix = elv_flat.reshape(LAT.shape) # z is currently a 1D, but we will reshape it into the 2D coordinate form - recognizing that we want it to follow the shape of X.
c1 = ax.pcolormesh(LONG, LAT, elv_matrix, cmap = 'viridis')
fig.colorbar(c1)
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Depth (m)')
ax.set_title('Initial conditions')


fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.plot_surface(LONG,LAT,elv_matrix)
ax.set_title('inital conditions')
ax.set_xlabel('x')
ax.set_zlabel('z')
ax.set_ylabel('y')

### ---------- A MATRIX AND RUNNING THE MODEL
"""
I did not do this part because my test dataset was much too big. Try to keep the number of x and y nodes to 100 - so if your study area is 1km x 1km, then you will want to use a 10m or 30m elevation dataset rather than lidar. If you are using lidar, your study area should be 100m x 100m (or similar).
"""

A = np.zeros ((n_lat*n_long, n_lat*n_long))

for i in range(n_lat): 
    for k in range(n_long):
        ik = i*n_long + k
        ### ---boundary conditions ---
        if i == 0: 
            A[ik,ik] = 1 ## no change 
        elif i ==1:
            A[ik,ik] = 1
       
        elif i == (n_lat-1): 
            A[ik,ik]=1
        elif i == (n_lat-2): 
            A[ik,ik]=1
        elif k == 0: 
            A[ik,ik]=1
        elif k == 1: 
           A[ik,ik]=1
        elif k == (n_long-1):
            A[ik,ik]=1
        elif k == (n_long-2):
           A[ik,ik]=1
        else:
            ##--- matrix coefficient time---
            A[ik,ik] = 1-2*sx -2*sy 
            A[ik,(i+1)*n_long + k] = sx 
            A[ik,(i-1)*n_long + k] = sx 
            A[ik, i*n_long +k + 1] = sy
            A[ik, i*n_long +k - 1] = sy
##print(A)
   
### ----- RUN A QUCK CHANGE -----
#elv_matrix += np.random.random((LAT.shape))*5 # 5 meter random additions

## Run Matrix 
print(elv_flat)
time = 0 
totaltime = 1000 #years 
while time <= totaltime: 
    new_z = np.dot (A,elv_flat)
    elv_flat[:] = new_z*1
    time += dt

print(elv_flat)
## plots final conditions 

fig, ax = plt.subplots(1,1) # use this method
elv_matrix = elv_flat.reshape(LAT.shape) # z is currently a 1D, but we will reshape it into the 2D coordinate form - recognizing that we want it to follow the shape of X.
c1 = ax.pcolormesh(LONG, LAT, elv_matrix, cmap = 'viridis')
fig.colorbar(c1)
ax.set_xlabel('Distance (m)')
ax.set_ylabel('distance  (m)')
ax.set_title('Final Consitions')


fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.plot_surface(LONG,LAT,elv_matrix)
ax.set_title('Final Conditions')
ax.set_xlabel('x')
ax.set_zlabel('z')
ax.set_ylabel('y')




### ---- SAVE ASCII OUTPUT (this can be opened in qgis easily!) -----
header = 'NCOLS %s \n' % n_long + 'NROWS %s \n' % n_lat + 'xllcorner %s \n' % xllcorner+ 'yllcorner %s \n' % yllcorner + 'cellsize %s \n' % dxy + 'NODATA_value -9999'
np.savetxt('new_elev.asc', elv_matrix, header = header, comments = '')
    
