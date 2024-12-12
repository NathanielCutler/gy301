#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  11 12:28:06 2024

@author: sschanz
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

### -------- INITIAL CONDITIONS -------------

# import .asc data
ascii_grid = np.loadtxt("Jungle_creek.asc", dtype = 'float', skiprows=6)

ascii_headers = np.loadtxt("Jungle_creek.asc", max_rows = 6, dtype = 'str')
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

## ---- DIFFUSION
D = 0.004 # m2/year

## ---- INITIAL ELEVATION
elv_flat = ascii_grid.flatten()


### --------- PLOT INITIAL CONDITIONS -------------

fig, ax = plt.subplots(1,1) # use this method
elv_matrix = elv_flat.reshape(LONG.shape) # z is currently a 1D, but we will reshape it into the 2D coordinate form - recognizing that we want it to follow the shape of X.
c1 = ax.pcolormesh(LONG, LAT, elv_matrix, cmap = 'viridis')
fig.colorbar(c1)
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Depth (m)')
ax.set_title('Initial conditions')

### ---------- A MATRIX AND RUNNING THE MODEL
"""
I did not do this part because my test dataset was much too big. Try to keep the number of x and y nodes to 100 - so if your study area is 1km x 1km, then you will want to use a 10m or 30m elevation dataset rather than lidar. If you are using lidar, your study area should be 100m x 100m (or similar).
"""
   
### ----- RUN A QUCK CHANGE -----
elv_matrix += np.random.random((LAT.shape))*5 # 5 meter random additions


### ---- SAVE ASCII OUTPUT (this can be opened in qgis easily!) -----
header = 'NCOLS %s \n' % n_long + 'NROWS %s \n' % n_lat + 'xllcorner %s \n' % xllcorner+ 'yllcorner %s \n' % yllcorner + 'cellsize %s \n' % dxy + 'NODATA_value -9999'
np.savetxt('new_elev.asc', elv_matrix, header = header, comments = '')
    
