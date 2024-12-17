# 2D model of Hillslope Diffusion from a DEM in ASCII format 
This model is designed to take an ASCII file and model hillslope diffusion on that topography 

## Inputs 
you will need an ASCII file with elevation data and you will need to set the run time, dt, and diffusion rate. The Ascii on this repo is from a 10m dem from the USGS Data downloader 
## Outputs 
this model will return 3 plots: the initial topography, the topography after the model run, and the elevation differences between the two. 
It will also output a new ASCII file with the elevation data for the post-run topography. 
