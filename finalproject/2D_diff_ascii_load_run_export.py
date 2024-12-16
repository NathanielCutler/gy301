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
ascii_grid = np.loadtxt("gy301_3dprint.asc", dtype = 'float', skiprows=6) ### pulls the data from the ascii file, skipping the headings to give an array of elevation data 

ascii_grid [ascii_grid <0] = np.nan## gets rid of holes in the elevation data 
ascii_grid = ascii_grid[1:-1,1:-1] ## trims the boundary of the data to remove anomolies 


ascii_headers = np.loadtxt("gy301_3dprint.asc", max_rows = 6, dtype = 'str') ##defines headers
n_lat, n_long = ascii_grid.shape ## establishes the scale of the grid based on the ascii file
dxy = ascii_headers[4,1].astype(float)  ## determines the scale of dxy from the ascii
xllcorner = ascii_headers[2,1].astype(float) ## identifies the location of the x corner
yllcorner = ascii_headers[3,1].astype(float) ## identifies the location of the y corner

## ---- GRID FORMATION
x = np.arange(0, dxy*n_lat, dxy) + xllcorner # array of x values
y = np.arange(0, dxy*n_long, dxy) + yllcorner # array of z values
LAT, LONG = np.meshgrid(x, y, indexing='ij') # this sets up a plotting grid
nodes = n_long*n_lat

## ---- TIME STEP and RUN TIME
dt = 10 # years, change this if run time is too long for desired totaltime
time = 0 
totaltime = 1000 #years 
dx = dxy #meters 
dy = dxy #meters, grid is based on square cells in this case so they will be the same

## ---- DIFFUSION
D = 0.004 # m2/year, standard for semiarid alluvium, environment dependant 

Dx = D
Dy = D

## ---- INITIAL ELEVATION

elv_flat = ascii_grid.flatten() ## flattens in order to be able to run matrix dot product 



### stability conditions (Von Neumann)

sx = dt * Dx / dx**2

sy = dt * Dy / dy**2

import sys 
##stop the run if it will be unstable
if sx>0.5: 
    print ("x is unstable")
    sys.exit() 
elif sy > 0.5: 
    print("y is unstable")
    sys.exit()
    

### --------- PLOT INITIAL CONDITIONS -------------

elv_matrix_pre = elv_flat.reshape(LAT.shape) # z is currently a 1D, but we will reshape it into the 2D coordinate form - recognizing that we want it to follow the shape of X.


## establish the slopes of the surface in order to plot by slope 
Z_x, Z_y = np. gradient (elv_matrix_pre, dxy, dxy)
slope = np.sqrt (Z_x **2 + Z_y **2)
slope_normalized = slope/(np.max(slope)) ##normalizes slope to make gradient more clear in plots  

##plots the figures 

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
surf = ax.plot_surface((LONG-np.min(LONG)),(LAT-np.min(LAT)),elv_matrix_pre, facecolors=plt.cm.viridis(slope_normalized),  # Colormap based on slope
                       rstride=1, cstride=1, cmap='viridis', linewidth=0, antialiased=True )
ax.set_title('Current topography of North Fork Henson Creek')
ax.set_xlabel('Distance (m)', labelpad=10)
ax.set_zlabel('elevation(m)', labelpad=10)
ax.set_ylabel('Distance (m)', labelpad=10)

#establish the scale/colorbar for the color gradient 
cbar = fig.colorbar(surf, location = 'left', shrink = 0.6)
cbar.set_label('Magnitude of normalized slope (slope/(max slope)')

### ---------- A MATRIX AND RUNNING THE MODEL

A = np.zeros ((n_lat*n_long, n_lat*n_long)) ##establish scale of A matrix 

for i in range(n_lat): 
    for k in range(n_long):
        ik = i*n_long + k
        ### ---boundary conditions ---
        if i == 0: 
            A[ik,ik] = 1 ## no change 
       
        elif i == (n_lat-1): 
            A[ik,ik]=1

        elif k == 0: 
            A[ik,ik]=1
   
        elif k == (n_long-1):
            A[ik,ik]=1
     
        else:
            ##--- matrix coefficient time---
            A[ik,ik] = 1-2*sx -2*sy 
            A[ik,(i+1)*n_long + k] = sx 
            A[ik,(i-1)*n_long + k] = sx 
            A[ik, i*n_long +k + 1] = sy
            A[ik, i*n_long +k - 1] = sy


## Run Matrix 


A_m = np.ma.masked_invalid(A)  # Mask invalid (NaN or Inf) values in matrix A
elv_flat_m = np.ma.masked_invalid(elv_flat)  # Mask invalid values in elevation data

##creates a boolean condition, assuming mask and data cleaning worked, should print false
print(np.any(np.isnan(A_m)))  # Check for NaNs in A_m
print(np.any(np.isnan(elv_flat_m)))  # Check for NaNs in elv_flat_m


while time<=totaltime:
    newz = np.ma.dot(A_m,elv_flat_m)
    elv_flat_m[:]= newz 
    time += dt


## plots final conditions 

elv_matrix = elv_flat_m.reshape(LAT.shape) # z is currently a 1D, but we will reshape it into the 2D coordinate form - recognizing that we want it to follow the shape of X.

## establish the slopes of the surface in order to plot by slope 
Z_x, Z_y = np. gradient (elv_matrix, dxy, dxy)
slope = np.sqrt (Z_x **2 + Z_y **2)
slope_normalized = slope/(np.max(slope)) ##normalizes slope to make gradient more clear in plots   
 

## plot the figures 
fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
surf = ax.plot_surface((LONG-np.min(LONG)),(LAT-np.min(LAT)),elv_matrix, facecolors=plt.cm.viridis(slope_normalized),  # Colormap based on slope
                       rstride=1, cstride=1, cmap='viridis', linewidth=0, antialiased=True )
ax.set_title('Topography of North Fork Henson Creek with only diffusion for ' + str(totaltime) + 'yrs')
ax.set_xlabel('Distance (m)', labelpad=10)
ax.set_zlabel('Elevation (meters)', labelpad=10)
ax.set_ylabel('Distance (m)', labelpad=10)

cbar = fig.colorbar(surf, location = 'left', shrink = 0.6)
cbar.set_label('Magnitude of normalized slope (slope/(max slope)')


elv_diff = elv_matrix_pre-elv_matrix ## establishes a matrix of how much the elevation at each node changed

##plots said matrix
fig = plt.figure()
ax1 = fig.add_subplot(111,projection = '3d')
ax1.plot_surface((LONG-np.min(LONG)),(LAT-np.min(LAT)),elv_diff,cmap = "RdBu")
ax1.set_title("Elevation Difference")
ax1.set_xlabel('Distance (m)', labelpad=10)
ax1.set_zlabel('Elevation (meters)', labelpad=10)
ax1.set_ylabel('Distance (m)', labelpad=10)

### ---- SAVE ASCII OUTPUT (this can be opened in qgis easily!) -----
header = 'NCOLS %s \n' % n_long + 'NROWS %s \n' % n_lat + 'xllcorner %s \n' % xllcorner+ 'yllcorner %s \n' % yllcorner + 'cellsize %s \n' % dxy + 'NODATA_value -9999'
np.savetxt('new_elev.asc', elv_matrix, header = header, comments = '')
    
